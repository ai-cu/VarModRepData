#process 4 images in a row and based on expectation change the boundary
# [n1] -> [n1] -> [n2] -> [n2]
import processor as pr

expected = []

def process_QPSK_image_stream(img_signals, expected):
    constellation = [1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]
    bin_data = []
    exp_ct = 0
    exp = expected
    for symbol in img_signals:
        closest_symbol = min(constellation, key=lambda x: abs(x - symbol))

        # Find expected index
        exp_sym = exp[exp_ct]

        # Decide between 10% reduced distance to expected vs observed symbol 
        if exp_sym != closest_symbol:
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

# Main function
if __name__ == "__main__":
    ip_1 = "data/n1.png"
    bd_1 = pr.image_to_bits(ip_1)
    qs_1 = pr.qpsk_modulation(bd_1)

    ip_2 = "data/n2.png"
    bd_2 = pr.image_to_bits(ip_2)
    qs_2 = pr.qpsk_modulation(bd_2)

    # First processing
    exp1, bin_data1 = process_QPSK_image_stream(qs_1, expected)

    # Second processing
    exp2, bin_data2 = process_QPSK_image_stream(qs_1, exp1)

    # Third processing
    exp3, bin_data3 = process_QPSK_image_stream(qs_2, exp2)

    # Fourth processing
    exp4, bin_data4 = process_QPSK_image_stream(qs_2, exp3)
