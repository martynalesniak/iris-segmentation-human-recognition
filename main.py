
import cv2
import numpy as np
import matplotlib.pyplot as plt
from operations import binarization, erosion, dilatation, horizontal_projection, vertical_projection

def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Konwersja do RGB
    return image

def process_image(image_path, kernel_size=5):
    # Załaduj obraz
    image = load_image(image_path)

    # Zdefiniowanie czynników binaryzacji dla tęczówki i źrenicy
    factor_iris = 2.5  # Dobierz eksperymentalnie dla tęczówki
    factor_pupil = 3.0  # Dobierz eksperymentalnie dla źrenicy

    # Zastosowanie binaryzacji na obrazie dla tęczówki
    binary_iris = binarization(image, factor=factor_iris)

    # Zastosowanie binaryzacji na obrazie dla źrenicy
    binary_pupil = binarization(image, factor=factor_pupil)

    # Zdefiniowanie jądra do operacji morfologicznych (np. kwadratowe jądro 5x5)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Wykonanie erozji i dylatacji w celu oczyszczenia obrazu
    eroded_image_iris = erosion(binary_iris, kernel)
    dilated_image_iris = dilatation(eroded_image_iris, kernel)

    # Wykonanie erozji i dylatacji dla źrenicy
    eroded_image_pupil = erosion(binary_pupil, kernel)
    dilated_image_pupil = dilatation(eroded_image_pupil, kernel)

    # Wykrywanie środków źrenicy i tęczówki za pomocą projekcji
    center_y_iris, radius_y_iris = horizontal_projection(dilated_image_iris, factor=factor_iris)
    center_x_iris, radius_x_iris = vertical_projection(dilated_image_iris, factor=factor_iris)

    center_y_pupil, radius_y_pupil = horizontal_projection(dilated_image_pupil, factor=factor_pupil)
    center_x_pupil, radius_x_pupil = vertical_projection(dilated_image_pupil, factor=factor_pupil)

    # Średni promień
    radius_iris = (radius_y_iris + radius_x_iris) // 2
    radius_pupil = (radius_y_pupil + radius_x_pupil) // 2

    # Rysowanie prostokąta wokół tęczówki
    iris_top_left = (center_x_iris - radius_iris, center_y_iris - radius_iris)
    iris_bottom_right = (center_x_iris + radius_iris, center_y_iris + radius_iris)

    # Rysowanie prostokąta wokół źrenicy
    pupil_top_left = (center_x_pupil - radius_pupil, center_y_pupil - radius_pupil)
    pupil_bottom_right = (center_x_pupil + radius_pupil, center_y_pupil + radius_pupil)

    # Za pomocą OpenCV rysujemy prostokąty wokół tęczówki i źrenicy
    image_with_rect = cv2.rectangle(image.copy(), iris_top_left, iris_bottom_right, (0, 255, 0), 2)
    image_with_rect = cv2.rectangle(image_with_rect, pupil_top_left, pupil_bottom_right, (255, 0, 0), 2)

    # Wyświetlanie obrazu z zaznaczoną tęczówką i źrenicą
    plt.imshow(image_with_rect)
    plt.title("Wykryta tęczówka i źrenica")
    plt.axis('off')
    plt.show()

    # Zapisanie wynikowego obrazu
    result_path = 'detected_iris_pupil.jpg'
    cv2.imwrite(result_path, cv2.cvtColor(image_with_rect, cv2.COLOR_RGB2BGR))
    print(f"Zapisano obraz z zaznaczoną tęczówką i źrenicą: {result_path}")


if __name__ == '__main__':
    process_image('teczowka_data/002/IMG_002_L_2.JPG')
