import subprocess
import time
import platform


def load_world():
    # Full path to the Webots executable
    if platform.system() == "macOS":
        webots_path = "/Applications/Webots.app/Contents/MacOS/webots"
    elif platform.system() == "Windows":
        webots_path = "C:\Program Files\Webots\lib\controller\python"
    else:
        webots_path = "C:\Program Files\Webots\lib\controller\python" # change for linux


    # List of available world files
    world_files = {
        '1': 'worlds/circle.wbt',
        '2': 'worlds/door.wbt'
    }

    # Continuously loop through the world files
    while True:
        for key, world_file in world_files.items():
            # Build the command to open Webots with the specified world
            command = [webots_path, "--mode=fast", world_file]

            # Start the process
            process = subprocess.Popen(command)

            # Wait for 5 seconds while the map is loaded
            time.sleep(5)

            # Terminate the process to close the map
            process.terminate()

            # Optionally wait for the process to ensure it has been cleaned up
            process.wait()

            print(f"Closed {world_file} and loading next...")


if __name__ == '__main__':
    load_world()
