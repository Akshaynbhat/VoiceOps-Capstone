import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_folder")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()
    raise NotImplementedError("Connect Gnani STT -> agent -> Gnani TTS.")

if __name__ == "__main__":
    main()
