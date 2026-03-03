import mlflow
from kedro.framework.hooks import hook_impl


class MLflowHook:
    @hook_impl
    def before_pipeline_run(self, run_params, pipeline, catalog):
        print(">>> MLflow hook triggered")
        mlflow.set_tracking_uri("./mlruns")
        mlflow.set_experiment("audio_prediction")
        mlflow.autolog()
        mlflow.start_run()
        print(f">>> Active run: {mlflow.active_run().info.run_id}")

    @hook_impl
    def after_pipeline_run(self, run_params, pipeline, catalog):
        if mlflow.active_run():
            mlflow.end_run()

    @hook_impl
    def on_pipeline_error(self, error, run_params, pipeline, catalog):
        if mlflow.active_run():
            mlflow.end_run(status="FAILED")