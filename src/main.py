import logging
import os
import mlflow

os.makedirs('Logs', exist_ok=True)
logging.basicConfig(
    filename=os.path.join('Logs', 'running.log'),
    format="[%(asctime)s: %(module)s: %(levelname)s]: %(message)s",
    level=logging.INFO,
    filemode='w'
)

def main():
    logging.info('Starting the application...')
    with mlflow.start_run() as run:
        mlflow.run('.', '', use_conda=False)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
        logging.exception(e)
        raise e