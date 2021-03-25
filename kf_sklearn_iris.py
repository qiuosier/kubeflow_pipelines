from pickle import load
import kfp
from kfp import dsl
from kfp.components import func_to_container_op, InputPath, OutputPath


def load_and_split(
    train_x_path: OutputPath(str), 
    test_x_path: OutputPath(str),
    train_y_path: OutputPath(str), 
    test_y_path: OutputPath(str)
):
    import numpy
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    iris_x, iris_y = datasets.load_iris(return_X_y=True)
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


def test_knn(test_x_path: InputPath(), test_y_path: InputPath(), model_path: InputPath()):
    import numpy
    import pickle
    from sklearn.metrics import classification_report
    test_x = numpy.load(test_x_path)
    test_y = numpy.load(test_y_path)
    print(f"Testing data size - X: {test_x.size}, y: {test_y.size}")
    with open(model_path, 'rb') as f:
        knn = pickle.load(f)
    pred_y = knn.predict(test_x)
    print(classification_report(test_y, pred_y))


def train_test_knn():
    load_op = func_to_container_op(load_and_split, base_image='qiuosier/sklearn')
    train_op = func_to_container_op(train_knn, base_image='qiuosier/sklearn')
    test_op = func_to_container_op(test_knn, base_image='qiuosier/sklearn')
    
    load_task = load_op()
    train_task = train_op(load_task.outputs['train_x'], load_task.outputs['train_y'])
    test_task = test_op(load_task.outputs['test_x'], load_task.outputs['test_y'], train_task.output)


@dsl.pipeline(name='sklearn-iris', description='Classifying Iris data with KNN.')
def kf_pipeline():
    train_test_knn()


# if __name__ == '__main__':
#     # Compile the pipeline
#     kfp.compiler.Compiler().compile(kf_pipeline, __file__ + '.yaml')
