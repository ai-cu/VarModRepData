#process 4 images in a row and based on expectation change the boundary
# [n1] -> [n1] -> [n2] -> [n2]
import processor as pr
import random

def process_simple(img_signals):
    constellation = [1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]
    bin_data = []
    for symbol in img_signals:
        closest_symbol = min(constellation, key=lambda x: abs(x - symbol))

        symbol_index = constellation.index(closest_symbol)
        bin_data.extend([int(bit) for bit in format(symbol_index, '02b')])
    return bin_data

def process_QPSK_image_stream_LRU(img_signals, expected):
    constellation = [1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]
    bin_data = []
    exp_ct = 0
    exp = expected.copy()
    for symbol in img_signals:
        closest_symbol = min(constellation, key=lambda x: abs(x - symbol))

        # Find expected index
        if exp_ct < len(exp): exp_sym = exp[exp_ct] 
        else: exp_sym = None

        # Decide between 10% reduced distance to expected vs observed symbol 
        if exp_sym != closest_symbol and exp_sym != None:
            dist_to_exp = abs(exp_sym - symbol)
            dist_to_clos = abs(closest_symbol - symbol)
            if 0.9 * dist_to_exp <= dist_to_clos:
                closest_symbol = exp_sym

        # Update exp
        if len(exp) != len(img_signals):
            exp.append(closest_symbol)
        else:
            exp[exp_ct] = closest_symbol

        exp_ct+=1

        symbol_index = constellation.index(closest_symbol)
        bin_data.extend([int(bit) for bit in format(symbol_index, '02b')])
    return (exp, bin_data)

def process_QPSK_image_stream_PROB(img_signals, expected, prob):
    constellation = [1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]
    bin_data = []
    exp_ct = 0
    exp = expected.copy()
    for symbol in img_signals:
        closest_symbol = min(constellation, key=lambda x: abs(x - symbol))

        # Find expected index
        if exp_ct < len(exp): exp_sym = exp[exp_ct] 
        else: exp_sym = None

        # Decide between 10% reduced distance to expected vs observed symbol 
        if exp_sym != closest_symbol and exp_sym != None:
            dist_to_exp = abs(exp_sym - symbol)
            dist_to_clos = abs(closest_symbol - symbol)
            if 0.9 * dist_to_exp <= dist_to_clos:
                closest_symbol = exp_sym

        # Update exp
        if len(exp) != len(img_signals):
            exp.append(closest_symbol)
        else:
            random_number = random.random()
            if random_number <= prob:
                exp[exp_ct] = closest_symbol

        exp_ct+=1

        symbol_index = constellation.index(closest_symbol)
        bin_data.extend([int(bit) for bit in format(symbol_index, '02b')])

    return (exp, bin_data)

def comp_bin_data(d1, d2):
    if len(d1) != len(d2):
        raise ValueError
    else:
        ct = 0
        for i in range(len(d1)):
            if d1[i] != d2[i]:
                ct += 1
        return ct/len(d1)
 
# Main function
if __name__ == "__main__":
    channel_SNR = 3

    ip_1 = "data/n2.png"
    bd_1 = pr.image_to_bits(ip_1)
    qs_1 = pr.qpsk_modulation(bd_1)
    ns_11 = pr.add_noise(qs_1, channel_SNR)
    ns_12 = pr.add_noise(qs_1, channel_SNR)

    ip_2 = "data/n3.png"
    bd_2 = pr.image_to_bits(ip_2)
    qs_2 = pr.qpsk_modulation(bd_2)
    ns_21 = pr.add_noise(qs_2, channel_SNR)
    ns_22 = pr.add_noise(qs_2, channel_SNR)

    # First processing
    exp1, bin_data1 = process_QPSK_image_stream_LRU(ns_11, [])

    # Second processing
    exp2, bin_data2 = process_QPSK_image_stream_LRU(ns_12, exp1)

    # Third processing
    exp3, bin_data3 = process_QPSK_image_stream_LRU(ns_21, exp2)

    # Fourth processing
    exp4, bin_data4 = process_QPSK_image_stream_LRU(ns_22, exp3)

    # First processing
    exp12, bin_data12 = process_QPSK_image_stream_PROB(ns_11, [], 0.5)

    # Second processing
    exp22, bin_data22 = process_QPSK_image_stream_PROB(ns_12, exp12, 0.5)

    # Third processing
    exp32, bin_data32 = process_QPSK_image_stream_PROB(ns_21, exp22, 0.5)

    # Fourth processing
    exp42, bin_data42 = process_QPSK_image_stream_PROB(ns_22, exp32, 0.5)

    # Check the differences?
    # LRU
    print("Accuracy post demodulation with simple statistical model - LRU, compared to demodulation without inference")
    print(comp_bin_data(bin_data1, bd_1))
    print(comp_bin_data(process_simple(ns_11), bd_1))
    print(comp_bin_data(bin_data2, bd_1))
    print(comp_bin_data(process_simple(ns_12), bd_1))
    print(comp_bin_data(bin_data3, bd_2))
    print(comp_bin_data(process_simple(ns_21), bd_2))
    print(comp_bin_data(bin_data4, bd_2))
    print(comp_bin_data(process_simple(ns_22), bd_2))
    
    print("Difference in progressing expectation lists")
    print(comp_bin_data(exp1, exp2))
    print(comp_bin_data(exp3, exp2))
    print(comp_bin_data(exp4, exp3))

    # Probabilistic substitution
    print("Accuraccy post demodulation with simple statistical model - Probabilistic Substitution, compared to demodulation without inference")
    print(comp_bin_data(bin_data12, bd_1))
    print(comp_bin_data(process_simple(ns_11), bd_1))
    print(comp_bin_data(bin_data22, bd_1))
    print(comp_bin_data(process_simple(ns_12), bd_1))
    print(comp_bin_data(bin_data32, bd_2))
    print(comp_bin_data(process_simple(ns_21), bd_2))
    print(comp_bin_data(bin_data42, bd_2))
    print(comp_bin_data(process_simple(ns_22), bd_2))
    
    print("Difference in progressing expectation lists")
    print(comp_bin_data(exp12, exp22))
    print(comp_bin_data(exp22, exp32))
    print(comp_bin_data(exp32, exp42))