import matplotlib.pyplot as plt
import seaborn as sns
import cv2 as cv


def change_width(ax, new_value):
    for patch in ax.patches:
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)


def create_graph_all():
    #CLASSES = ["Train", "figure1", "covid_chestxray", "covid19_radiography", "actualmed", "rsna"]
    CLASSES = ["Train", "figure1", "chestxray", "radiography",
               "actualmed", "rsna"]  # número de gráficos
    '''
    FIGURE1 = [0.8611, 1.0, 0.69565, 0.30848, 0.31578, 0.0]
    CHESTXRAY = [0.8657, 1.0, 0.78260, 0.37017, 0.31578, 0.10423]
    RADIOGRAPHY = [0.9874, 0.5, 0.60869, 0.94601, 0.47368, 0.31540]
    ACTUALMED = [0.9036, 1.0, 0.76, 0.75590, 0.94736, 0.86004]
    RSNA = [0.9585, 0.0, 0.23809, 0.18587, 0.0, 0.05110]

    TRAIN = [0.8611, 0.8657, 0.9874, 0.9036, 0.9585]
    FIGURE1 = [1.0, 1.0, 0.5, 1.0, 0.0]
    CHESTXRAY = [0.69565, 0.78260, 0.60869, 0.76, 0.23809]
    RADIOGRAPHY = [0.30848, 0.37017, 0.94601, 0.75590, 0.18587]
    ACTUALMED = [0.31578, 0.31578, 0.47368, 0.94736, 0.0]
    RSNA = [0.0, 0.10423, 0.31540, 0.86004, 0.05110]'''

    TRAIN = [0.8611, 0.8657, 0.9874, 0.9036, 0.9585]
    FIGURE1 = [1.0, 1.0, 0.5, 1.0, 0.0]
    CHESTXRAY = [0.69565, 0.78260, 0.60869, 0.76, 0.23809]
    RADIOGRAPHY = [0.30848, 0.37017, 0.94601, 0.75590, 0.18587]
    ACTUALMED = [0.31578, 0.31578, 0.47368, 0.94736, 0.0]
    RSNA = [0.0, 0.10423, 0.31540, 0.86004, 0.05110]

    """
    DB_NAMES = [
        "Figure 1 COVID-19 Chest X-ray",
        "COVID Chest X-Ray",
        "COVID-19 Radiography",
        "Actualmed COVID-19 Chest X-ray",
        "RSNA Pneumonia Detection Challenge"
    ]"""

    DB_NAMES = [
        "Model Figure 1",
        "Model COVID Chest X-Ray",
        "Model COVID-19 Radiography",
        "Model Actualmed",
        "Model RSNA"
    ]

    OUTPUT_NAMES = [
        "model_figure1",
        "model_covid_chestxray",
        "model_covid19_radiography",
        "model_actualmed",
        "model_rsna"
    ]

    for i in range(5):
        plt.rcParams['font.family'] = "serif"
        plt.rcParams['font.weight'] = "bold"
        plt.rcParams['figure.figsize'] = 13, 10
        sns.set_style('whitegrid')
        fig, ax = plt.subplots()

        barplot = sns.barplot(
            x=CLASSES, y=[TRAIN[i], FIGURE1[i], CHESTXRAY[i], RADIOGRAPHY[i], ACTUALMED[i], RSNA[i]])

        barplot.patches[0].set_color('forestgreen')
        barplot.patches[1].set_color('darkorange')
        barplot.patches[2].set_color('darkorange')
        barplot.patches[3].set_color('darkorange')
        barplot.patches[4].set_color('darkorange')
        barplot.patches[5].set_color('darkorange')

        barplot.patches[i+1].set_color('steelblue')
        change_width(ax, .4)

        font_size = 20
        plt.title(DB_NAMES[i], fontsize=font_size, fontweight="bold")
        plt.xlabel('Dataset', fontsize=font_size, fontweight="bold")
        plt.ylabel('Acurracy', fontsize=font_size, fontweight="bold")
        plt.ylim(0, 1)
        barplot.tick_params(labelsize=font_size)

        graph_name = "graph_" + str(OUTPUT_NAMES[i]) + ".png"
        plt.savefig(graph_name, pad_inches=0, bbox_inches='tight')

        img = cv.imread(graph_name)
        height, width, _ = img.shape
        height = HEIGHT  # int(height * 0.5)
        width = WIDTH  # int(width * 0.5)
        img = cv.resize(img, (width, height))
        cv.imwrite(graph_name, img)


WIDTH = 645
HEIGHT = 478

create_graph_all()
