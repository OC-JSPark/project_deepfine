import kfp
from kfp import dsl

def preprocess_op():
    return dsl.ContainerOp(
        name='Preprocess Data',
        image='bti8coc/preprocess:latest',
        file_outputs={
            'train_data': '/app/train_data.pt',
            'valid_data': '/app/valid_data.pt'
        }
    )

def train_op(train_data, valid_data):
    return dsl.ContainerOp(
        name='Train Model',
        image='bti8coc/train:latest',
        arguments=[
            '--train_data', train_data,
            '--valid_data', valid_data
        ],
        file_outputs={
            'model': '/app/model.pth'
        }
    ).after(preprocess_op)

def test_op(model):
    return dsl.ContainerOp(
        name='Test Model',
        image='bti8coc/test:latest',
        arguments=[
            '--model', model
        ],
        file_outputs={
            'metrics': '/app/metrics.txt'
        }
    ).after(train_op)

def deploy_op(model):
    return dsl.ContainerOp(
        name='Deploy Model',
        image='bti8coc/deploy:latest',
        arguments=[
            '--model', model
        ]
    ).after(test_op)

@dsl.pipeline(
   name='NLP Pipeline',
   description='An example pipeline that trains and deploys an NLP model.'
)
def nlp_pipeline():
    _preprocess_op = preprocess_op()
    
    _train_op = train_op(
        dsl.InputArgumentPath(_preprocess_op.outputs['train_data']),
        dsl.InputArgumentPath
