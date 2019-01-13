import argparse
import os.path
from pkg.NaiveBayes import naive_bayes_classifier
from pkg.GaussianNaiveBayes import gaussian_bayes_classifier


def is_valid_arguments(training_file: str, test_file: str, algorithm: str, supports: []) -> bool:
    if os.path.isfile(training_file) and os.path.isfile(test_file) and algorithm in supports:
        return True
    else:
        return False


def main():
    # initialize system
    supported_algorithms = ['naive', 'gaussian']

    parser = argparse.ArgumentParser()

    parser.add_argument('-tr', '--training', action='store', dest='training_data', required=True,
                        help='The training data sample of Bayes classifier algorithm')

    parser.add_argument('-te', '--test', action='store', dest='test_data', required=True,
                        help='The test data sample of Bayes classifier algorithm')

    parser.add_argument('-c', '--class', action='store', dest='class_variable', required=True,
                        help='The class variable of prediction or output for each row of feature matrix')

    parser.add_argument('-i', '--index', action='store_false', default=False,
                        dest='is_indexed',
                        help='Whether the training data sample and test data sample has index column')

    parser.add_argument('-o', '--output', action='store', required=True,
                        dest='output_file',
                        help='The test data sample after applying given Bayes classifier algorithm')

    parser.add_argument('-p', '--positive', action='store', required=True, dest='positive_values', nargs='+',
                        default=[],
                        help='Positive values of class variable')

    parser.add_argument('-n', '--negative', action='store', required=True, dest='negative_values', nargs='+',
                        default=[],
                        help='Negative values of class variable')

    parser.add_argument('-a', '--algorithm', action='store', dest='bayes_algorithm',
                        default='naive',
                        help='Naive or Gaussian Bayes Algorithm')

    parser.add_argument('-v', '--version', action='version',
                        version='%(prog)s 1.0')

    results = parser.parse_args()
    print('training_data     = {!r}'.format(results.training_data))
    print('test_data   = {!r}'.format(results.test_data))
    print('class_variable        = {!r}'.format(results.class_variable))
    print('is_indexed        = {!r}'.format(results.is_indexed))
    print('output_file        = {!r}'.format(results.output_file))
    print('positive_values        = {!r}'.format(results.positive_values))
    print('negative_values        = {!r}'.format(results.negative_values))
    print('algorithm        = {!r}'.format(results.bayes_algorithm))

    if is_valid_arguments(results.training_data, results.test_data, results.bayes_algorithm, supported_algorithms):
        if results.bayes_algorithm == 'naive':
            naive_bayes_classifier(results.training_data,
                                   results.test_data,
                                   results.output_file,
                                   results.class_variable,
                                   results.positive_values,
                                   results.negative_values,
                                   results.is_indexed)
        else:
            gaussian_bayes_classifier(results.training_data,
                                      results.test_data,
                                      results.output_file,
                                      results.class_variable,
                                      results.is_indexed)

    else:
        parser.error('Invalid options provided')
        exit(1)


if __name__ == '__main__':
    main()
