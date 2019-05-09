from nlgeval import compute_metrics
metrics_dict = compute_metrics(hypothesis='../dataset/tran.txt',
                               references=['../dataset/ref.txt'])
print(metrics_dict)

# or: nlg-eval --no-glove --hypothesis=../code/k/ml-base/pku-deep-learning/lg-course/assignment4/dataset/tran.txt --references=../code/k/ml-base/pku-deep-learning/lg-course/assignment4/dataset/ref.txt