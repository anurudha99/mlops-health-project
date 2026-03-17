from src.data.models.train import train

def run_pipeline():

    print("Starting training pipeline...")

    train()

    print("Pipeline finished")


if __name__ == "__main__":
    run_pipeline()