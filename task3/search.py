import collections
import json

all_pages = {num for num in range(1, 101)}


def load_index(json_filename):
    with open(json_filename, 'r') as json_file:
        data = json.load(json_file)
    inverted_index = {}
    for item in data:
        inverted_index[item['word']] = set(item['inverted_array'])
    return inverted_index


def tokenize_query(query):
    query = query.replace('(', '( ')
    query = query.replace(')', ' )')
    query = query.split(' ')
    return query


def convert_to_postfix(infix_tokens):
    precedence = {}
    precedence['NOT'] = 3
    precedence['AND'] = 2
    precedence['OR'] = 1
    precedence['('] = 0
    precedence[')'] = 0

    output = []
    operator_stack = []

    for token in infix_tokens:
        if token == '(':
            operator_stack.append(token)
        elif token == ')':
            operator = operator_stack.pop()
            while operator != '(':
                output.append(operator)
                operator = operator_stack.pop()
        elif token in precedence:
            if operator_stack:
                current_operator = operator_stack[-1]
                while operator_stack and precedence[current_operator] > precedence[token]:
                    output.append(operator_stack.pop())
                    if operator_stack:
                        current_operator = operator_stack[-1]
            operator_stack.append(token)
        else:
            output.append(token.lower())

    while operator_stack:
        output.append(operator_stack.pop())
    return output


def bool_and(set1, set2):
    return set1.intersection(set2)


def bool_or(set1, set2):
    return set1.union(set2)


def bool_not(set1):
    return all_pages.difference(set1)


# Перед выполнением нужно запустить task1/main.py для скачивания файлов и task3/main.py для создания индекса
# Пример запроса: (hello OR world) AND NOT often
if __name__ == '__main__':
    index = load_index('inverted_index.txt')
    query = tokenize_query(input('Enter query: '))
    results_stack = []
    token_queue = collections.deque(convert_to_postfix(query))
    try:
        while token_queue:
            token = token_queue.popleft()
            if token != 'AND' and token != 'OR' and token != 'NOT':
                results_stack.append(index[token] if token in index else set())
            elif token == 'AND':
                set1 = results_stack.pop()
                set2 = results_stack.pop()
                results_stack.append(bool_and(set1, set2))
            elif token == 'OR':
                set1 = results_stack.pop()
                set2 = results_stack.pop()
                results_stack.append(bool_or(set1, set2))
            elif token == 'NOT':
                set1 = results_stack.pop()
                results_stack.append(bool_not(set1))
        result = results_stack.pop()
        print(result if result else 'No results found.')
    except IndexError:
        print('Error: invalid query.')
