import os

if __name__ == '__main__':
    data_path = './data/idioms.txt'
    data = [line.strip() for line in open(data_path, 'r')]
    data = sorted(data, key=lambda x: len(x), reverse=True)
    with open('./data/idioms.txt', 'w', encoding='utf-8') as wf:
        for value in data:
            wf.write(value + '\n')
