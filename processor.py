import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_ML_model_pred(img_signals, expected, constellation):
    #look at prediction vs observed and do the 10% thing
    bin_data = []
    ml_pred = expected[0]
    ml_signals = [constellation[min(round(x), 63)] for x in ml_pred]
    exp_ct = 0
    for symbol in img_signals:
        closest_symbol = min(constellation, key=lambda x: abs(x - symbol))

        # Find expected index
        exp_sym = ml_signals[exp_ct] 

        # Decide between 10% reduced distance to expected vs observed symbol 
        if exp_sym != closest_symbol and exp_sym != None:
            dist_to_exp = abs(exp_sym - symbol)
            dist_to_clos = abs(closest_symbol - symbol)
            if 0.9 * dist_to_exp <= dist_to_clos:
                closest_symbol = exp_sym

        exp_ct+=1

        symbol_index = constellation.index(closest_symbol)
        bin_data.append(symbol_index)
    return bin_data

# Function to convert an image to binary data
def image_to_bits(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read the image
    
    binary_data = np.unpackbits(img)  # Convert image to binary data
    return binary_data

# Function to modulate binary data using QPSK
def qam4_modulation(binary_data):
    # Define 4-QAM constellation points
    # The points are arranged in a square grid.
    constellation = {
        (0, 0): (-1-1j),  # Example constellation point for 00
        (0, 1): (-1+1j),  # Example constellation point for 01
        (1, 0): (1-1j),   # Example constellation point for 10
        (1, 1): (1+1j)    # Example constellation point for 11
    }

    # Ensure the input bit array is a multiple of 2
    if len(binary_data) % 2 != 0:
        raise ValueError("Number of bits must be a multiple of 2 for 4-QAM")

    # Split the bit array into chunks of 2 bits
    bit_chunks = [binary_data[i:i+2] for i in range(0, len(binary_data), 2)]

    # Map each chunk to a constellation point
    symbols = [constellation[tuple(chunk)] for chunk in bit_chunks]

    return symbols

def qam16_modulation(binary_data):
    # Define 16 QAM constellation points
    constellation = {
        (0, 0, 0, 0): (-3+3j), (0, 0, 0, 1): (-3+1j), 
        (0, 0, 1, 0): (-3-3j), (0, 0, 1, 1): (-3-1j), 
        (0, 1, 0, 0): (-1+3j), (0, 1, 0, 1): (-1+1j), 
        (0, 1, 1, 0): (-1-3j), (0, 1, 1, 1): (-1-1j), 
        (1, 0, 0, 0): (3+3j),  (1, 0, 0, 1): (3+1j), 
        (1, 0, 1, 0): (3-3j),  (1, 0, 1, 1): (3-1j), 
        (1, 1, 0, 0): (1+3j),  (1, 1, 0, 1): (1+1j), 
        (1, 1, 1, 0): (1-3j),  (1, 1, 1, 1): (1-1j)
    }

    bit_chunks = [binary_data[i:i+4] for i in range(0, len(binary_data), 4)]

    # Map each chunk to a constellation point
    symbols = [constellation[tuple(chunk)] for chunk in bit_chunks]

    return symbols

def qam64_modulation(binary_data):
    # Define 64-QAM constellation points
    # Each point is represented by a unique combination of amplitude and phase.
    # Constellation points are typically arranged in a square grid.
    constellation = {}
    values = [-7, -5, -3, -1, 1, 3, 5, 7]  # Possible values for I and Q components
    bit_combinations = [(i, j, k, l, m, n) for i in [0, 1] for j in [0, 1] for k in [0, 1] for l in [0, 1] for m in [0, 1] for n in [0, 1]]

    for i, combination in enumerate(bit_combinations):
        I = values[i // 8]  # Integer division to cycle through I values
        Q = values[i % 8]   # Modulus to cycle through Q values
        constellation[combination] = complex(I, Q)

    # Ensure the input bit array is a multiple of 6
    if len(binary_data) % 6 != 0:
        raise ValueError("Number of bits must be a multiple of 6 for 64-QAM")

    # Split the bit array into chunks of 6 bits
    bit_chunks = [binary_data[i:i+6] for i in range(0, len(binary_data), 6)]

    # Map each chunk to a constellation point
    symbols = [constellation[tuple(chunk)] for chunk in bit_chunks]

    return symbols

# Function to add complex Gaussian noise to the QPSK symbols
def add_noise(qpsk_symbols, snr_dB):
    # Calculate the signal power from QPSK symbols
    signal_power = np.mean(np.abs(qpsk_symbols) ** 2)

    # Calculate noise power from SNR (in dB)
    snr = 10 ** (snr_dB / 10)
    noise_power = signal_power / snr

    # Generate complex Gaussian noise
    noise = np.sqrt(noise_power / 2) * (np.random.randn(len(qpsk_symbols)) + 1j * np.random.randn(len(qpsk_symbols)))

    # Add noise to the QPSK symbols
    noisy_symbols = qpsk_symbols + noise

    return noisy_symbols

def symbol_stream_diff(symbols1, symbols2):
    # Counting the number of different pixels
    diff = 0
    
    for i in range(len(symbols1)):
        # If two symbols unequal, add to counter
        if symbols1[i] != symbols2[i]:
            diff += 1
    
    # Return ratio of different symbols
    return diff/len(symbols1)

# Main function
if __name__ == "__main__":
    # Specify the path to the image
    image_path_1 = "data/simple2.png"

    # Convert the image to binary data
    binary_data_1 = image_to_bits(image_path_1)

    # Modulate the binary data using QPSK
    qpsk_symbols_1 = qam4_modulation(binary_data_1)

    # Add noise to the QPSK symbols (e.g., SNR of 10 dB)
    snr_dB = 25
    noisy_symbols = add_noise(qpsk_symbols_1, snr_dB)

    ip_2 = "B76C2C61-80FA-48C9-957E-B9AEEAD9740D_1_105_c.jpeg"
    bd_2 = image_to_bits(ip_2)
    qs_2 = qam4_modulation(bd_2)

    qs_3 = qam64_modulation(bd_2)
    noisy_symbols_2 = add_noise(qs_3, snr_dB)

    print(symbol_stream_diff(qpsk_symbols_1, qs_2))

    plt.scatter(np.real(noisy_symbols_2), np.imag(noisy_symbols_2), marker='o', label='16QAM Symbols')
    plt.title("16QAM Constellation Diagram")
    plt.xlabel("Real")
    plt.ylabel("Imaginary")
    plt.grid()
    plt.legend()
    plt.show()