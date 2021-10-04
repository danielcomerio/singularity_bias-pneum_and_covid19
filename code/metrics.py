from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, recall_score, classification_report
import numpy as np
import argparse
import os


def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-inf", "--information",
                        help="path to the information directory", type=str)

    args = parser.parse_args()

    return args


def main():
    args = parse_command_line_args()

    FOLDERS_NAME = [
        "model_figure1",
        "model_covid_chestxray",
        "model_covid19_radiography",
        "model_actualmed",
        "model_rsna"
    ]

    for folder_name in FOLDERS_NAME:
        for file_name in FOLDERS_NAME:
            file_tests = os.path.join(
                args.information, folder_name, "tests", folder_name + "-IN-" + file_name + ".txt")

            file_tests = open(file_tests, "r")
            # file_tests.write("path_imagem, classe_real, classe_predita")

            label_list = []
            predict_list = []
            line = file_tests.readline()
            while line:
                line = line.strip().split(", ")
                label_list.append(line[1])
                prediction = [float(line[2]), float(line[3]), float(line[4])]
                prediction = np.argmax(prediction, axis=-1)
                predict_list.append(str(prediction))

                line = file_tests.readline()

            file_tests.close()

            file_metrics = os.path.join(
                args.information, folder_name, "metrics")

            if file_name == FOLDERS_NAME[0]:
                os.makedirs(file_metrics)

            file_metrics = os.path.join(
                file_metrics, folder_name + "-IN-" + file_name + ".txt")
            file_metrics = open(file_metrics, "w")

            file_metrics.write("Model used: " + folder_name + '\n')
            file_metrics.write("Dataset tested: " + file_name[6:] + "\n\n")

            file_metrics.write("confusion_matrix: \n" + str(confusion_matrix(
                label_list, predict_list)) + '\n')
            file_metrics.write("Accuracy: " + str(accuracy_score(
                label_list, predict_list)) + '\n')
            file_metrics.write("Precision: " + str(precision_score(
                label_list, predict_list, average='macro')) + '\n')
            file_metrics.write("Recall: " + str(recall_score(
                label_list, predict_list, average='macro')) + '\n')
            file_metrics.write(
                "F1-score: " + str(f1_score(label_list, predict_list, average='macro')) + '\n')

            file_metrics.close()


if __name__ == "__main__":
    main()
