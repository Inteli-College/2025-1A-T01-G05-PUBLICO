import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
import psutil
import time
import sys
from rtabmap_msgs.msg import Info

# --- Importação segura de bibliotecas de GPU ---
NVIDIA_LIB_AVAILABLE = False
JETSON_LIB_AVAILABLE = False

try:
    import pynvml
    try:
        pynvml.nvmlInit()
        NVIDIA_LIB_AVAILABLE = True
    except pynvml.NVMLError:
        print("Biblioteca pynvml encontrada, mas não foi possível inicializar.", file=sys.stderr)
except ImportError:
    pass

try:
    from jtop import jtop
    JETSON_LIB_AVAILABLE = True
except ImportError:
    pass


class PerformanceMonitorNode(Node):
    def __init__(self):
        super().__init__('performance_monitor_node')
        
        # --- Lógica de Controle: Iniciar imediatamente ---
        self.start_time = time.time()
        self.total_duration = 0.0

        # Listas para armazenar as medições
        self.cpu_usage = []
        self.ram_usage_mb = []
        self.rtabmap_fps = []
        
        # --- Para medição de GPU ---
        self.gpu_usage = []
        self.vram_usage_mb = []
        self.gpu_handle = None
        self.jetson_stats = None
        self.hardware_type = self.detect_hardware()

        if self.hardware_type == "nvidia_pc":
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        elif self.hardware_type == "jetson":
            self.jetson_stats = jtop()
            self.jetson_stats.start()

        # Subscriber para pegar o FPS diretamente do RTAB-Map
        self.info_sub = self.create_subscription(Info, '/rtabmap/info', self.rtabmap_info_callback, 10)

        # O Timer de coleta é criado e iniciado imediatamente
        self.monitor_timer = self.create_timer(1.0, self.collect_metrics)

        self.get_logger().info(f"🚀 Monitor de performance iniciado. Hardware: {self.hardware_type}.")
        self.get_logger().info("Coletando dados... Pressione Ctrl+C para parar e gerar o relatório.")

    def detect_hardware(self):
        if JETSON_LIB_AVAILABLE and 'aarch64' in sys.platform:
            return "jetson"
        if NVIDIA_LIB_AVAILABLE:
            return "nvidia_pc"
        return "cpu_only"

    def get_gpu_metrics(self):
        if self.hardware_type == "nvidia_pc" and self.gpu_handle:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            return util.gpu, mem.used / (1024**2)
        elif self.hardware_type == "jetson" and self.jetson_stats and self.jetson_stats.ok():
            gpu_util = self.jetson_stats.gpu['val']
            ram_used = self.jetson_stats.ram['use'] / 1024
            return gpu_util, ram_used
        return None, None

    def rtabmap_info_callback(self, msg):
        # A única responsabilidade desta função é coletar o FPS
        try:
            key_to_find = "Timing/Total/ms"
            if key_to_find in msg.stats_keys:
                idx = msg.stats_keys.index(key_to_find)
                total_time_ms = msg.stats_values[idx]
                if total_time_ms > 0:
                    frequency = 1000.0 / total_time_ms
                    self.rtabmap_fps.append(frequency)
        except Exception as e:
            self.get_logger().error(f"Erro ao processar /rtabmap/info: {e}")

    def collect_metrics(self):
        # Coleta as métricas a cada segundo, sem condição de parada interna
        cpu_percent = psutil.cpu_percent()
        ram_mb = psutil.virtual_memory().used / (1024**2)
        self.cpu_usage.append(cpu_percent)
        self.ram_usage_mb.append(ram_mb)

        log_msg = f"CPU: {cpu_percent:.1f}%, RAM: {ram_mb:.0f} MB"
        
        gpu_percent, vram_mb = self.get_gpu_metrics()
        if gpu_percent is not None:
            self.gpu_usage.append(gpu_percent)
            self.vram_usage_mb.append(vram_mb)
            gpu_label = "VRAM" if self.hardware_type == "nvidia_pc" else "Shared RAM"
            log_msg += f", GPU: {gpu_percent:.1f}%, {gpu_label}: {vram_mb:.0f} MB"
        
        self.get_logger().info(log_msg)

    def stop_and_summarize(self):
        # Calcula a duração total no momento em que é chamado
        self.total_duration = time.time() - self.start_time
        self.get_logger().info(f"\n--- Medição interrompida após {self.total_duration:.2f}s. Gerando resumo... ---")
        
        if self.monitor_timer:
            self.monitor_timer.cancel()

        avg_cpu = sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0
        max_cpu = max(self.cpu_usage) if self.cpu_usage else 0
        avg_ram = sum(self.ram_usage_mb) / len(self.ram_usage_mb) if self.ram_usage_mb else 0
        max_ram = max(self.ram_usage_mb) if self.ram_usage_mb else 0
        avg_fps = sum(self.rtabmap_fps) / len(self.rtabmap_fps) if self.rtabmap_fps else 0
        
        summary = f"""
        --- 📊 Relatório Final de Performance (Duração: {self.total_duration:.2f}s) ---
        
        ⏱️ Eficiência (Real-Time Factor):
          - Frequência Média do RTAB-Map: {avg_fps:.2f} Hz

        🔋 Consumo de Recursos:
          CPU:
            - Média: {avg_cpu:.2f}%
            - Pico:  {max_cpu:.2f}%
          Memória RAM (Total do Sistema):
            - Média: {avg_ram:.2f} MB
            - Pico:  {max_ram:.2f} MB
        """
        
        if self.gpu_usage:
            avg_gpu = sum(self.gpu_usage) / len(self.gpu_usage) if self.gpu_usage else 0
            max_gpu = max(self.gpu_usage) if self.gpu_usage else 0
            avg_vram = sum(self.vram_usage_mb) / len(self.vram_usage_mb) if self.vram_usage_mb else 0
            max_vram = max(self.vram_usage_mb) if self.vram_usage_mb else 0
            gpu_label = "VRAM" if self.hardware_type == "nvidia_pc" else "Shared RAM"
            summary += f"""
          GPU ({self.hardware_type}):
            - Utilização Média: {avg_gpu:.2f}%
            - Utilização de Pico: {max_gpu:.2f}%
          Memória de Vídeo ({gpu_label}):
            - Média: {avg_vram:.2f} MB
            - Pico:  {max_vram:.2f} MB
        """

        summary += "\n        ------------------------------------------"
        self.get_logger().info(summary)

    def __del__(self):
        if self.hardware_type == "nvidia_pc" and NVIDIA_LIB_AVAILABLE:
            pynvml.nvmlShutdown()
        if self.hardware_type == "jetson" and self.jetson_stats:
            self.jetson_stats.close()

def main(args=None):
    rclpy.init(args=args)
    monitor_node = PerformanceMonitorNode()
    
    try:
        rclpy.spin(monitor_node)
    except KeyboardInterrupt:
        monitor_node.get_logger().info("Ctrl+C detectado.")
    finally:
        # A sequência de parada correta, acionada pelo Ctrl+C
        monitor_node.stop_and_summarize()
        monitor_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()