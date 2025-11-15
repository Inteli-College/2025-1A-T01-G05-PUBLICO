import freenect
import time
import signal
import sys
import subprocess
import shutil

def initialize_kinect():
    """
    Executa o 'freenect-micview' como um subprocesso para forçar a
    inicialização completa do hardware do Kinect, incluindo o firmware de áudio.
    """
    micview_path = shutil.which('freenect-wavrecord')
    if not micview_path:
        print("ERRO: O executável 'freenect-wavrecord' não foi encontrado no seu PATH.")
        print("Certifique-se de que o libfreenect foi instalado corretamente (ex: /usr/local/bin).")
        return False

    print("Inicializando o hardware do Kinect (executando 'freenect-wavrecord' por 2s)...")
    try:
        subprocess.run(
            ['timeout', '2s', micview_path],
            check=True,
            capture_output=True,
            text=True
        )
        print("Hardware inicializado com sucesso.")
        return True
    except subprocess.CalledProcessError as e:
        if e.returncode == 124:
            print("Hardware inicializado (timeout esperado).")
            return True
        else:
            print(f"Erro ao tentar inicializar o Kinect com 'freenect-micview':")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
            return False
    except FileNotFoundError:
        print("ERRO: O comando 'timeout' não foi encontrado. Instale com 'sudo apt install coreutils'.")
        return False

keep_running = True

def handler(signum, frame):
    global keep_running
    print("\nSinal de parada recebido, desligando...")
    keep_running = False

signal.signal(signal.SIGINT, handler)


if __name__ == "__main__":
    if not initialize_kinect():
        sys.exit(1)

    print("\nIniciando captura de vídeo e profundidade...")
    print("Pressione Ctrl+C para parar.")

    try:
        while keep_running:
            depth, timestamp_d = freenect.sync_get_depth()
            video, timestamp_v = freenect.sync_get_video()

            if depth is None or video is None:
                print("Frame perdido! Tentando novamente...")
                continue

            video = video[:, :, ::-1]
            print(f"Frame recebido! Vídeo: {video.shape}, Profundidade: {depth.shape} em {timestamp_v}")

            time.sleep(0.03)

    except KeyboardInterrupt:
        print("Interrupção do teclado detectada.")
    finally:
        print("Parando o loop síncrono...")
        freenect.sync_stop()
        print("Programa finalizado.")