# Use the official AWS Lambda Python base image
FROM public.ecr.aws/lambda/python:3.10

# Copy requirements.txt and install dependencies
COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install -r requirements.txt

# Copy the function code
COPY models/pokemon_model.pkl ${LAMBDA_TASK_ROOT}
COPY models/pokemon_preprocessor.pkl ${LAMBDA_TASK_ROOT}
COPY datasets/pokemon_complete_dataset.csv ${LAMBDA_TASK_ROOT}
COPY models/save_preprocessing_pipeline.py ${LAMBDA_TASK_ROOT}
COPY src/utils.py ${LAMBDA_TASK_ROOT}
COPY tests/test_handler.py ${LAMBDA_TASK_ROOT}
COPY src/lambda_function.py ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler (function) name
CMD ["lambda_function.lambda_handler"]
