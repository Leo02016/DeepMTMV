from sklearn.model_selection import StratifiedKFold
import os
from bs4 import BeautifulSoup as Soup
import codecs
import numpy as np
import nltk
import random
import stat


class Load_datasets():
    def __init__(self, stage, num_classes):
        self.class_num = num_classes
        if stage == 'train' or stage == 'Train':
            view_1 = np.load('./data/train_web_content.npy')
            view_2 = np.load('./data/train_web_links.npy')
            view_3 = np.load('./data/train_web_title.npy')
            self.data = [view_1, view_2, view_3]
            self.label = np.load('./data/train_label.npy').astype(int) - 1
        elif stage == 'test' or stage == 'Test':
            view_1 = np.load('./data/test_web_content.npy')
            view_2 = np.load('./data/test_web_links.npy')
            view_3 = np.load('./data/test_web_title.npy')
            self.data = [view_1, view_2, view_3]
            self.label = np.load('./data/test_label.npy').astype(int) - 1
        else:
            raise (NameError('The stage should be either train or test'))
            
    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        view_1 = self.data[0][idx]
        view_2 = self.data[1][idx]
        view_3 = self.data[2][idx]
        y = self.label[idx]
        sample = {'view_1': np.array(view_1), 'view_2': np.array(view_2), 'view_3': np.array(view_3), 'label': y}
        return sample


def preprocess(data_path):
    word2vec_size = 30
    k_fold = 5
    task = ['cornell', 'washington', 'texas', 'wisconsin']
    feature_type = ['course', 'department', 'faculty', 'project', 'student', 'staff']
    data_path = data_path
    web_content = []
    web_title = []
    web_links = []
    label = []
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    tok_func = lambda x: [el.lower() for el in tokenizer.tokenize(x)]
    for file in os.listdir(data_path):
        if file in feature_type:
            os.chdir('%s/%s' % (data_path, file))
            print('%s/%s' % (data_path, file))
            for subfile in os.listdir(os.getcwd()):
                if subfile in task:
                    file_count = 0
                    os.chdir('%s/%s/%s' % (data_path, file, subfile))
                    print('%s/%s/%s' % (data_path, file, subfile))
                    for url in os.listdir(os.getcwd()):
                        try:
                            with codecs.open(url, "rU", encoding='utf-8', errors='ignore') as fdata:
                                soup = Soup(fdata, "html.parser")
                                file_count += 1
                                # ---- get title of the web page ---
                                title = soup.title
                                if title is None:
                                    title = []
                                else:
                                    title = [tok_func(x) for x in title.text.split() if tok_func(x) != []]
                                    new_title = []
                                    for sublist in title:
                                        if len(sublist) != 0:
                                            for item in sublist:
                                                new_title.append(item)
                                        else:
                                            new_title.append(sublist)
                                    title = new_title
                                # ---- get all words in the web page ---
                                text = soup.get_text()
                                text = text.split()
                                tokens = [tok_func(x) for x in text if tok_func(x) != []]
                                new_tokens = []
                                for x in tokens:
                                    new_tokens = new_tokens + x
                                #  ---- get all links pointing to that page  ----
                                links = []
                                for link in soup.find_all(name="a"):
                                    if 'href' in link.attrs:
                                        text = link.text
                                        text = text.split()
                                        text = [tok_func(x) for x in text if len(x) != 0 and tok_func(x) != []]
                                        for x in text:
                                            links.append(x)
                                New_links = []
                                for sublist in links:
                                    for item in sublist:
                                        New_links.append(item)
                                links = New_links
                                # add features
                                if subfile == 'cornell':
                                    web_content.append(new_tokens)
                                    web_links.append(links)
                                    web_title.append(title)
                                    if file == 'course':
                                        label.append(1)
                                    else:
                                        label.append(2)
                                elif subfile == 'washington':
                                    web_content.append(new_tokens)
                                    web_links.append(links)
                                    web_title.append(title)
                                    if file == 'course':
                                        label.append(3)
                                    else:
                                        label.append(4)
                                elif subfile == 'texas':
                                    web_content.append(new_tokens)
                                    web_links.append(links)
                                    web_title.append(title)
                                    if file == 'course':
                                        label.append(5)
                                    else:
                                        label.append(6)
                                elif subfile == 'wisconsin':
                                    web_content.append(new_tokens)
                                    web_links.append(links)
                                    web_title.append(title)
                                    if file == 'course':
                                        label.append(7)
                                    else:
                                        label.append(8)
                        except OSError:
                            print("File open Error!")
    os.chdir('%s' % data_path)
    os.chdir('../../')

    web_content = np.array(web_content)
    web_links = np.array(web_links)
    web_title = np.array(web_title)
    label = np.array(label)
    # splitting data into training data set and testing data set
    skf = StratifiedKFold(n_splits=k_fold, random_state=10, shuffle=True)
    for train_index, test_index in skf.split(np.zeros(len(label)), label):
        random.shuffle(train_index)
        random.shuffle(test_index)
        train_web_content = web_content[train_index]
        test_web_content = web_content[test_index]
        train_web_links = web_links[train_index]
        test_web_links = web_links[test_index]
        train_web_title = web_title[train_index]
        test_web_title = web_title[test_index]
        train_label = label[train_index]
        test_label = label[test_index]
        break
    print('Saving label...')
    np.save('./data/train_label.npy', np.array(train_label))
    np.save('./data/test_label.npy', np.array(test_label))

    # save the key words extracted from web content in a txt file
    f = open('./data/web_content.txt', 'w+')
    for i in range(len(train_web_content)):
        for j in range(len(train_web_content[i])):
            f.write(train_web_content[i][j]+' ')
        f.write('\n')
    f.close()
    # save the key words extracted from web link in a txt file
    print('\n')
    f = open('./data/web_links.txt', 'w+')
    for i in range(len(train_web_links)):
        for j in range(len(train_web_links[i])):
            f.write(train_web_links[i][j] + ' ')
        f.write('\n')
    f.close()
    # save the key words extracted from web title in a txt file
    print('\n')
    f = open('./data/web_title.txt', 'w+')
    for i in range(len(train_web_title)):
        for j in range(len(train_web_title[i])):
            f.write(train_web_title[i][j] + ' ')
        f.write('\n')
    f.close()

    # word2vec representation
    if not os.path.exists('./data/web_content_representation.txt'):
        os.chmod('./word2vec_representation.sh', stat.S_IMODE(os.lstat('./word2vec_representation.sh')[stat.ST_MODE]) | 751)
        os.system('./word2vec_representation.sh')
    dictionary = dict()
    with open('./data/web_content_representation.txt', 'r') as f:
        # ignore the first line, since the first line is the shape of the matrix
        next(f)
        for line in f:
            line = line.rstrip()
            listA = line.split(' ')
            dictionary[listA[0]] = list(map(float, listA[1:]))
    f.close()
    max_length = 0
    for i in range(len(train_web_content)):
        if max_length < len(train_web_content[i]):
            max_length = len(train_web_content[i])
    print('\ndimensionality of web_content feature is {}'.format(max_length))
    matrix = np.zeros((len(train_web_content), max_length, word2vec_size))
    # for every line in tokens
    for i in range(len(train_web_content)):
        # for every word in every line
        for j in range(len(train_web_content[i])):
            word = train_web_content[i][j]
            # check if this word in dictionary
            if word in dictionary:
                matrix[i][j] = dictionary[word]
    web_content = matrix
    matrix = np.zeros((len(test_web_content), max_length, word2vec_size))
    # for every line in tokens
    for i in range(len(test_web_content)):
        # for every word in every line
        for j in range(len(test_web_content[i])):
            # in case of a situation where the feature dimensionality of test_data > train_data.
            if j >= len(train_web_content[0]):
                break
            word = test_web_content[i][j]
            # check if this word in dictionary
            if word in dictionary:
                matrix[i][j] = dictionary[word]
    test_web_content = matrix

    dictionary = dict()
    with open('./data/web_links_representation.txt', 'r') as f:
        # ignore the first line, since the first line is the shape of the matrix
        next(f)
        for line in f:
            line = line.rstrip()
            listA = line.split(' ')
            dictionary[listA[0]] = list(map(float, listA[1:]))
    f.close()
    max_length = 0
    for i in range(len(train_web_links)):
        if max_length < len(train_web_links[i]):
            max_length = len(train_web_links[i])
    print('\ndimensionality of web_links feature is {}'.format(max_length))
    matrix = np.zeros((len(train_web_links), max_length, word2vec_size))
    # for every line in tokens
    for i in range(len(train_web_links)):
        # for every word in every line
        for j in range(len(train_web_links[i])):
            word = train_web_links[i][j]
            # check if this word in dictionary
            if word in dictionary:
                matrix[i][j] = dictionary[word]
    web_links = matrix
    matrix = np.zeros((len(test_web_links), max_length, word2vec_size))
    # for every line in tokens
    for i in range(len(test_web_links)):
        # for every word in every line
        for j in range(len(test_web_links[i])):
            # in case of a situation where the feature dimensionality of test_data > train_data.
            if j >= len(train_web_links[0]):
                break
            word = test_web_links[i][j]
            # check if this word in dictionary
            if word in dictionary:
                matrix[i][j] = dictionary[word]
    test_web_links = matrix

    dictionary = dict()
    with open('./data/web_title_representation.txt', 'r') as f:
        # ignore the first line, since the first line is the shape of the matrix
        next(f)
        for line in f:
            line = line.rstrip()
            listA = line.split(' ')
            dictionary[listA[0]] = list(map(float, listA[1:]))
    f.close()
    max_length = 0
    for i in range(len(train_web_title)):
        if max_length < len(train_web_title[i]):
            max_length = len(train_web_title[i])
    print('\ndimensionality of web_title feature is {}'.format(max_length))
    matrix = np.zeros((len(train_web_title), max_length, word2vec_size))
    # for every line in tokens
    for i in range(len(train_web_title)):
        # for every word in every line
        for j in range(len(train_web_title[i])):
            word = train_web_title[i][j]
            # check if this word in dictionary
            if word in dictionary:
                matrix[i][j] = dictionary[word]
    web_title = matrix
    matrix = np.zeros((len(test_web_title), max_length, word2vec_size))
    # for every line in tokens
    for i in range(len(test_web_title)):
        # for every word in every line
        for j in range(len(test_web_title[i])):
            # in case of a situation where the feature dimensionality of test_data > train_data.
            if j >= len(train_web_title[0]):
                break
            word = test_web_title[i][j]
            # check if this word in dictionary
            if word in dictionary:
                matrix[i][j] = dictionary[word]
    test_web_title = matrix
    print('Saving three views...')
    np.save('./data/train_web_content.npy', np.array(web_content))
    np.save('./data/train_web_links.npy', np.array(web_links))
    np.save('./data/train_web_title.npy', np.array(web_title))
    np.save('./data/test_web_content.npy', np.array(test_web_content))
    np.save('./data/test_web_links.npy', np.array(test_web_links))
    np.save('./data/test_web_title.npy', np.array(test_web_title))
