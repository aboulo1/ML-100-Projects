from data_pipeline import DataPipeline
from model_pipeline import ModelPipeline

def main():
    """
    A quick function to train, evaluate and save the model in a few lines
    All the preprocessing is handled in the pipelines

    Returns
    -------
    None.

    """
    data_pipeline = DataPipeline()
    X_train, X_valid, _, y_train, y_valid = data_pipeline.get_data()
    # Model Pipeline
    model_pipeline = ModelPipeline()
    model_pipeline.train(X_train, y_train)
    accuracy = model_pipeline.evaluate(X_valid, y_valid)
    print(f"Model Accuracy: {accuracy:.2f}")
    model_pipeline.save_model()



if __name__ == "__main__":
    main()