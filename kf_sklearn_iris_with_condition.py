import kfp
from kfp import dsl
from kfp.components import func_to_container_op, InputPath, OutputPath


def load_data(x_path: OutputPath(str), y_path: OutputPath(str)):
    import numpy
    from sklearn import datasets
    iris_x, iris_y = datasets.load_iris(return_X_y=True)
    with open(x_path, 'wb') as f:
        numpy.save(f, iris_x)
    with open(y_path, 'wb') as f:
        numpy.save(f, iris_y)

def split_data(
    x_path: InputPath(), 
    y_path: InputPath(),
    train_x_path: OutputPath(str), 
    test_x_path: OutputPath(str),
    train_y_path: OutputPath(str), 
    test_y_path: OutputPath(str),
):
    import numpy
    from sklearn.model_selection import train_test_split
    iris_x = numpy.load(x_path)
    iris_y = numpy.load(y_path)
    train_x, test_x, train_y , test_y = train_test_split(iris_x, iris_y, test_size=0.2)
    print(f"Training data size - X: {train_x.size}, y: {train_y.size}")
    print(f"Testing data size - X: {test_x.size}, y: {test_y.size}")

    with open(train_x_path, 'wb') as f:
        numpy.save(f, train_x)
    with open(test_x_path, 'wb') as f:
        numpy.save(f, test_x)
    with open(train_y_path, 'wb') as f:
        numpy.save(f, train_y)
    with open(test_y_path, 'wb') as f:
        numpy.save(f, test_y)


def train_knn(train_x_path: InputPath(), train_y_path: InputPath(), model_path: OutputPath()):
    import numpy
    import pickle
    from sklearn.neighbors import KNeighborsClassifier
    train_x = numpy.load(train_x_path)
    train_y = numpy.load(train_y_path)
    print(f"Training data size - X: {train_x.size}, y: {train_y.size}")
    knn = KNeighborsClassifier()
    knn.fit(train_x, train_y)
    with open(model_path, 'wb') as f:
        pickle.dump(knn, f)

def train_logistics(train_x_path: InputPath(), train_y_path: InputPath(), model_path: OutputPath()):
    import numpy
    import pickle
    import random
    from sklearn.linear_model import LogisticRegression
    p = random.random()
    print(p)
    if p > 0.5:
        raise Exception()
    train_x = numpy.load(train_x_path)
    train_y = numpy.load(train_y_path)
    print(f"Training data size - X: {train_x.size}, y: {train_y.size}")
    knn = LogisticRegression()
    knn.fit(train_x, train_y)
    with open(model_path, 'wb') as f:
        pickle.dump(knn, f)


def test_model(test_x_path: InputPath(), test_y_path: InputPath(), model_path: InputPath()):
    import numpy
    import pickle
    import random
    from sklearn.metrics import classification_report
    p = random.random()
    print(p)
    if p > 0.5:
        raise Exception()
    test_x = numpy.load(test_x_path)
    test_y = numpy.load(test_y_path)
    print(f"Testing data size - X: {test_x.size}, y: {test_y.size}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    pred_y = model.predict(test_x)
    print(classification_report(test_y, pred_y))

@func_to_container_op
def select_classifier(minimum: int, maximum: int) -> int:
    """Generate a random number between minimum and maximum (inclusive)."""
    import random
    result = random.randint(minimum, maximum)
    print(result)
    return result

@func_to_container_op
def fail_op(message):
    """Fails."""
    import sys
    print(message)    
    sys.exit(1)

def train_test_knn():

    load_op = func_to_container_op(load_data, base_image='qiuosier/sklearn')
    split_op = func_to_container_op(split_data, base_image='qiuosier/sklearn')
    test_op = func_to_container_op(test_model, base_image='qiuosier/sklearn')
    train_knn_op = func_to_container_op(train_knn, base_image='qiuosier/sklearn')
    train_logistics_op = func_to_container_op(train_logistics, base_image='qiuosier/sklearn')
    
    load_task = load_op()

    split_task = split_op(load_task.outputs['x'], load_task.outputs['y'])
    split_task.execution_options.caching_strategy.max_cache_staleness = "P0D"

    p = select_classifier(0, 9)
    p.execution_options.caching_strategy.max_cache_staleness = "P0D"

    with dsl.Condition(p.output >= 5):
        train_task = train_knn_op(split_task.outputs['train_x'], split_task.outputs['train_y'])
        test_op(split_task.outputs['test_x'], split_task.outputs['test_y'], train_task.output)
    with dsl.Condition(p.output < 5):
        train_task = train_logistics_op(split_task.outputs['train_x'], split_task.outputs['train_y'])
        test_op(split_task.outputs['test_x'], split_task.outputs['test_y'], train_task.output)


@dsl.pipeline(name='sklearn-iris', description='Classifying Iris data with KNN.')
def kf_pipeline():
    train_test_knn()


# if __name__ == '__main__':
#     # Compile the pipeline
#     kfp.compiler.Compiler().compile(kf_pipeline, __file__ + '.yaml')
