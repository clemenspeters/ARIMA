FROM public.ecr.aws/lambda/python:3.10

ENV LAMBDA_TASK_ROOT=.

# Copy function code
COPY processor.py ${LAMBDA_TASK_ROOT}

# Install the function's dependencies using file requirements.txt
# from your project folder.

COPY requirements.txt .
RUN pip3 install -r requirements.txt -t ${LAMBDA_TASK_ROOT}
RUN pip3 install smart_open[http] -t ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "processor.handler" ]