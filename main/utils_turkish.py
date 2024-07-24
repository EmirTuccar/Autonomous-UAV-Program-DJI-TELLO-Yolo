import string
import easyocr
import csv

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=True)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5',
                    'B':'8',
                    'P':'3'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S',
                    '8':'B'}




def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    try:
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame_nmr', 'car_id', 'car_bbox',
                             'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                             'license_number_score'])

            for frame_nmr in results.keys():
                print(f"Processing frame {frame_nmr}")  # Debug statement
                for car_id in results[frame_nmr].keys():
                    print(f"Processing car {car_id} in frame {frame_nmr}")  # Debug statement
                    car_data = results[frame_nmr][car_id]
                    print(f"Car data: {car_data}")  # Debug statement
                    
                    if 'car' in car_data and 'license_plate' in car_data:
                        car_bbox = car_data['car'].get('bbox', [None, None, None, None])
                        license_plate = car_data['license_plate']
                        license_plate_bbox = license_plate.get('bbox', [None, None, None, None])
                        license_plate_bbox_score = license_plate.get('bbox_score', 'N/A')
                        license_number = license_plate.get('text', 'N/A')
                        license_number_score = license_plate.get('text_score', 'N/A')

                        writer.writerow([frame_nmr,
                                         car_id,
                                         '[{} {} {} {}]'.format(*car_bbox),
                                         '[{} {} {} {}]'.format(*license_plate_bbox),
                                         license_plate_bbox_score,
                                         license_number,
                                         license_number_score])
                    else:
                        print(f"Data missing for car {car_id} in frame {frame_nmr}")  # Debug statement
        print(f"Data successfully written to {output_path}")
    except Exception as e:
        print(f"Error writing to CSV: {e}")







def license_complies_format(text):
    """
    Check if the license plate text complies with the required format.

    Args:
        text (str): License plate text.

    Returns:
        bool: True if the license plate complies with the format, False otherwise.
    """
    if len(text) not in [7, 8]:
        return False

    if len(text) == 7:
        return (text[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[0] in dict_char_to_int.keys()) and \
               (text[1] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[1] in dict_char_to_int.keys()) and \
               (text[2] in string.ascii_uppercase or text[2] in dict_int_to_char.keys()) and \
               (text[3] in string.ascii_uppercase or text[3] in dict_int_to_char.keys()) and \
               (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
               (text[5] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[5] in dict_char_to_int.keys()) and \
               (text[6] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[6] in dict_char_to_int.keys())

    if len(text) == 8:
        return (text[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[0] in dict_char_to_int.keys()) and \
               (text[1] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[1] in dict_char_to_int.keys()) and \
               (text[2] in string.ascii_uppercase or text[2] in dict_int_to_char.keys()) and \
               (text[3] in string.ascii_uppercase or text[3] in dict_int_to_char.keys()) and \
               (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
               (text[5] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[5] in dict_char_to_int.keys()) and \
                   (text[6] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[6] in dict_char_to_int.keys()) and \
               (text[7] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[7] in dict_char_to_int.keys())

def format_license(text):
    """
    Format the license plate text by converting characters using the mapping dictionaries.

    Args:
        text (str): License plate text.

    Returns:
        str: Formatted license plate text.
    """
    license_plate_ = ''
    mapping = {0: dict_char_to_int, 1: dict_char_to_int, 2: dict_int_to_char, 3: dict_int_to_char,
               4: dict_int_to_char, 5: dict_char_to_int, 6: dict_char_to_int, 7: dict_char_to_int}

    for j in range(len(text)):
        if j in mapping and text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]

    return license_plate_



def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given cropped image.

    Args:
        license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.

    Returns:
        tuple: Tuple containing the formatted license plate text and its confidence score.
    """

    detections = reader.readtext(license_plate_crop)

    for detection in detections:
        bbox, text, score = detection

        text = text.upper().replace(' ', '')

        if license_complies_format(text):
            return format_license(text), score

    return None, None


def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.

    Args:
        license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.

    Returns:
        tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
    """
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1