import pickle
import numpy as np
import Levenshtein as ltn

target = 'welcome here'


def main():
    with open('pickle/classify200_original_and_adv.pickle', 'rb') as f:
        results,_ = pickle.load(f)
    # rates, total_rate = word_error_rate(results[0][:116], results[1][:116])
    success, total_success = success_rate(target, results[1][:116])
    # with open('pickle/classify_orginal_add_w_noise.pickle','rb') as f:
    #     result, length = pickle.load(f)
    #     results.append(result[0])
    # rates2, total_rate2 = word_error_rate(results[0],results[3])
    # print('result1:',total_rate)
    print('---------------')
    print('result2:',total_success)


def word_error_rate(target, x):
    error_rates = []
    for a,b in zip(target,x):
        a = string_operation(a)
        b = string_operation(b)
        a = a.split()
        b = b.split()
        count = 0
        length = len(a)
        for word in a:
            if word in b:
                count += 1
        error = abs(len(b)-length) + length - count
        error_rates.append(min(error/length, 1))

    error_rates = np.array(error_rates)
    return error_rates,error_rates.sum() / error_rates.size


def success_rate(target, x):
    x_len = len(x)
    success = np.zeros(x_len,dtype=np.int)
    for i in range(x_len):
        adv = x[i]
        adv = string_operation(adv)
        if adv == target:
            success[i] = 1
        else:
            a = target.split()
            b = adv.split()
            count = 0
            length = len(a)
            for word in a:
                if word in b:
                    count += 1
            success[i] = count / length

    return success, success.sum() / success.size


def string_operation(str):
    str = str.strip()
    return ' '.join(str.split())

main()