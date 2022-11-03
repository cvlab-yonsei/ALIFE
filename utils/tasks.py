tasks_voc = {
    "offline": {
        0: list(range(21)),
    },

    "19-1": {
        0: list(range(20)),
        1: [20],
    },

    "15-5": {
        0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        1: [16, 17, 18, 19, 20],
    },

    "hi": {
        0: [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19],
        1: [16, 18, 20],
        2: [1, 2],
    },

    "15-5s": {
        0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        1: [16],
        2: [17],
        3: [18],
        4: [19],
        5: [20]
    },
}


tasks_ade = {
    "offline": {
        0: [x for x in range(151)]
    },

    "100-50": {
        0: [x for x in range(0, 101)],
        1: [x for x in range(101, 151)]
    },

    "50": {
        0: [x for x in range(0, 51)],
        1: [x for x in range(51, 101)],
        2: [x for x in range(101, 151)]
    },

    "100-10": {
        0: [x for x in range(0, 101)],
        1: [x for x in range(101, 111)],
        2: [x for x in range(111, 121)],
        3: [x for x in range(121, 131)],
        4: [x for x in range(131, 141)],
        5: [x for x in range(141, 151)]
    },
}


def get_task_labels(dataset, name, step):
    if dataset == 'voc':
        task_dict = tasks_voc[name]
    elif dataset == 'ade':
        task_dict = tasks_ade[name]
    else:
        raise NotImplementedError
    assert step in task_dict.keys(), f"You should provide a valid step! [{step} is out of range]"

    labels = list(task_dict[step])
    labels_old = [label for s in range(step) for label in task_dict[s]]
    return labels, labels_old


def get_per_task_classes(dataset, name, step):
    if dataset == 'voc':
        task_dict = tasks_voc[name]
    elif dataset == 'ade':
        task_dict = tasks_ade[name]
    else:
        raise NotImplementedError
    assert step in task_dict.keys(), f"You should provide a valid step! [{step} is out of range]"

    classes = [len(task_dict[s]) for s in range(step + 1)]
    return classes


def get_tasks(dataset, task, step=None):
    if dataset == 'voc':
        tasks = tasks_voc
    elif dataset == 'ade':
        tasks = tasks_ade
    else:
        NotImplementedError
        
    if step is None:
        return tasks[task].copy()
    
    return tasks[task][step].copy()
