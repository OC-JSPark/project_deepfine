import kfp
from kfp import dsl

def preprocess_op():
    return dsl.ContainerOp(
        name='Preprocess Data',
        image='bti8coc/preprocess:latest',
        file_outputs={
            'subset_train_dataset': '/app/subset_train_dataset.pt',
            'subset_test_dataset': '/app/subset_test_dataset.pt'
        }
    )

def train_op(subset_train_dataset, subset_test_dataset):
    return dsl.ContainerOp(
        name='Train Model',
        image='bti8coc/train:latest',
        arguments=[
            '--subset_train_dataset', subset_train_dataset,
            '--subset_test_dataset', subset_test_dataset
        ],
        file_outputs={
            'model': '/app/models/vit_model.pth'
        }
    )

def test_op(model, subset_test_dataset):
    return dsl.ContainerOp(
        name='Test Model',
        image='bti8coc/test:latest',
        arguments=[
            '--model', model,
            '--subset_test_dataset', subset_test_dataset
        ],
        file_outputs={
            'metrics': '/app/metrics.txt'
        }
    )

def deploy_op(model):
    return dsl.ContainerOp(
        name='Deploy Model',
        image='bti8coc/deploy:latest',
        arguments=['--model', model]
    )

@dsl.pipeline(
    name='FashionMNIST Pipeline',
    description='An example pipeline that trains and deploys a FashionMNIST model.'
)
def fashionmnist_pipeline():
    preprocess_task = preprocess_op()
    
    train_task = train_op(
        subset_train_dataset=preprocess_task.outputs['subset_train_dataset'],
        subset_test_dataset=preprocess_task.outputs['subset_test_dataset']
    ).after(preprocess_task)
    
    test_task = test_op(
        model=train_task.outputs['model'],
        subset_test_dataset=preprocess_task.outputs['subset_test_dataset']
    ).after(train_task)
    
    deploy_task = deploy_op(
        model=train_task.outputs['model']
    ).after(test_task)

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(fashionmnist_pipeline, 'fashionmnist_pipeline.yaml')
