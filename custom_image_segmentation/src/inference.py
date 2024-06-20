import argparse

def main(args):
    print("Configuration File:", args.config)
    print("Trained Model Weights:", args.output)
    print("Image:", args.output)
    print("Device:", args.device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform inference with Detectron2")
    parser.add_argument("--config", required=True, help="Path to the configuration file")
    parser.add_argument("--output", required=False, help="Path to the output directory")
    parser.add_argument("--image", required=False, help="Path to the output directory")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Device to use for inference (default: cpu)")

    args = parser.parse_args()
    main(args)