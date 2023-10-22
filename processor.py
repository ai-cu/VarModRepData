import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to convert an image to binary data
def image_to_bits(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read the image
    binary_data = np.unpackbits(img)  # Convert image to binary data
    return binary_data

# Function to modulate binary data using QPSK
def qpsk_modulation(binary_data):
    # Define QPSK constellation points
    constellation = [1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]

    # Initialize variables
    symbols = []
    i = 0

    # Convert binary data to QPSK symbols
    while i < len(binary_data):
        if i + 1 < len(binary_data):
            # Take two bits at a time for QPSK modulation
            bits = binary_data[i:i+2]

            # Map the bits to a QPSK symbol
            symbol = constellation[int(''.join(map(str, bits)), 2)]
            symbols.append(symbol)

            i += 2
        else:
            break

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
    qpsk_symbols_1 = qpsk_modulation(binary_data_1)

    # Add noise to the QPSK symbols (e.g., SNR of 10 dB)
    snr_dB = 20
    noisy_symbols = add_noise(qpsk_symbols_1, snr_dB)

    ip_2 = "data/simple3.png"
    bd_2 = image_to_bits(ip_2)
    qs_2 = qpsk_modulation(bd_2)

    print(symbol_stream_diff(qpsk_symbols_1, qs_2))

    plt.scatter(np.real(noisy_symbols), np.imag(noisy_symbols), marker='o', label='QPSK Symbols')
    plt.title("QPSK Constellation Diagram")
    plt.xlabel("Real")
    plt.ylabel("Imaginary")
    plt.grid()
    plt.legend()
    plt.show()