# You need to learn a function with n inputs.
# For given number of inputs, we will generate random function.
# Your task is to learn it
import time
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ..utils import solutionmanager as sm
from ..utils.gridsearch import GridSearch

class SolutionModel(nn.Module):
    def __init__(self, input_size, output_size, solution):
        super(SolutionModel, self).__init__()
        if solution.grid_search.enabled:
            torch.manual_seed(solution.random)
        self.solution = solution
        self.input_size = input_size
        # sm.SolutionManager.print_hint("Hint[1]: Explore more deep neural networks")
        self.hidden_size = self.solution.hidden_size
        self.linear1 = nn.Linear(input_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear4 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear5 = nn.Linear(self.hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.solution.activations[self.solution.activation_hidden](x)
        x = self.linear2(x)
        x = self.solution.activations[self.solution.activation_hidden](x)
        x = self.linear3(x)
        x = self.solution.activations[self.solution.activation_hidden](x)
        x = self.linear4(x)
        x = self.solution.activations[self.solution.activation_hidden](x)
        x = self.linear5(x)
        x = torch.sigmoid(x)
        return x

    def calc_loss(self, output, target):
        loss = self.solution.loss_functions[self.solution.loss_function](output, target)
        return loss

    def calc_predict(self, output):
        predict = output.round()
        return predict

class Solution():
    def __init__(self):
        self.activations = {
            'sigmoid': nn.Sigmoid(),
            'relu': nn.ReLU(),
            'relu6': nn.ReLU6(),
            'rrelu0103': nn.RReLU(0.1, 0.3),
            'rrelu0205': nn.RReLU(0.2, 0.5),
            'htang1': nn.Hardtanh(-1, 1),
            'htang2': nn.Hardtanh(-2, 2),
            'htang3': nn.Hardtanh(-3, 3),
            'tanh': nn.Tanh(),
            'elu': nn.ELU(),
            'selu': nn.SELU(),
            'hardshrink': nn.Hardshrink(),
            'leakyrelu01': nn.LeakyReLU(0.1),
            'leakyrelu001': nn.LeakyReLU(0.01),
            'logsigmoid': nn.LogSigmoid(),
            'prelu': nn.PReLU(),
        }
        self.loss_functions = {
            'binary_cross_entropy': nn.BCELoss(),
            'binary_cross_entropy_with_logits': nn.BCEWithLogitsLoss(),
            'poisson_nll_loss': nn.PoissonNLLLoss(),
            # 'cosine_embedding_loss': nn.CosineEmbeddingLoss(),
            # 'cross_entropy': nn.CrossEntropyLoss(),
            # 'ctc_loss': nn.CTCLoss(),
            'hinge_embedding_loss': nn.HingeEmbeddingLoss(),
            'kl_div': nn.KLDivLoss(),
            'l1_loss': nn.L1Loss(),
            'mse_loss': nn.MSELoss(),
            # 'margin_ranking_loss': nn.MarginRankingLoss(),
            # 'multilabel_margin_loss': nn.MultiLabelMarginLoss(),
            'multilabel_soft_margin_loss': nn.MultiLabelSoftMarginLoss(),
            # 'multi_margin_loss': nn.MultiMarginLoss(),
            # 'nll_loss': nn.NLLLoss(),
            'smooth_l1_loss': nn.SmoothL1Loss(),
            'soft_margin_loss': nn.SoftMarginLoss(),
            # 'triplet_margin_loss': nn.TripletMarginLoss(),
        }
        self.learning_rate = 2.8
        self.momentum = 0.8
        self.hidden_size = 10
        self.activation_hidden = 'relu'
        self.loss_function = 'binary_cross_entropy'
        self.sols = {}
        self.solsSum = {}
        self.random = 3
        self.random_grid = [_ for _ in range(10)]
        # self.hidden_size_grid = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
        # self.hidden_size_grid = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        # self.learning_rate_grid = [0.1, 1.0, 2.0, 3.0, 5.0]
        # self.activation_hidden_grid = list(self.activations.keys())
        # self.activation_hidden_grid = list(self.activations.keys())
        # self.loss_function_grid = list(self.loss_functions.keys())
        self.grid_search = GridSearch(self)
        self.grid_search.set_enabled(False)


    def create_model(self, input_size, output_size):
        return SolutionModel(input_size, output_size, self)

    # Return number of steps used
    def train_model(self, model, train_data, train_target, context):
        step = 0
        # Put model in train mode
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        while True:
            time_left = context.get_timer().get_time_left()
            # No more time left, stop training
            key = "{}_{}_{}_{}_{}".format(self.learning_rate, self.momentum, self.hidden_size, self.activation_hidden, self.loss_function)
            # Speed up search
            if time_left < 0.1 or (self.grid_search.enabled and step > 1500):
                if not key in self.sols:
                    self.sols[key] = 0
                    self.solsSum[key] = 0
                self.sols[key] += 1
                self.solsSum[key] += step
                self.sols[key] = -1
                break
            if key in self.sols and self.sols[key] == -1:
                break
            data = train_data
            target = train_target
            # model.parameters()...gradient set to zero
            optimizer.zero_grad()
            # evaluate model => model.forward(data)
            # sm.SolutionManager.print_hint("Hint[2]: Explore other activation functions", step)
            output = model(data)
            # if x < 0.5 predict 0 else predict 1
            predict = model.calc_predict(output)
            # Number of correct predictions
            correct = predict.eq(target.view_as(predict)).long().sum().item()
            # Total number of needed predictions
            total = predict.view(-1).size(0)
            if correct == total:
                if not key in self.sols:
                    self.sols[key] = 0
                    self.solsSum[key] = 0
                self.sols[key] += 1
                self.solsSum[key] += step
                #if self.sols[key] > 1:
                #    print("Key = {} Avg = {:.2f} Ins = {}".format(key, float(self.solsSum[key])/self.sols[key], self.sols[key]))
                if self.sols[key] == len(self.random_grid):
                    # self.best_step = step
                    print("{:.4f}, Learning rate = {} Momentum = {} Hidden size = {} Activation1 hidden = {} Activation2 hidden = {} Loss function = {} Steps = {}".format(
                        float(self.solsSum[key]) / self.sols[key], self.learning_rate, self.momentum, self.hidden_size, self.hidden_size, self.activation_hidden, self.loss_function, step))
                break
            # calculate loss
            # sm.SolutionManager.print_hint("Hint[3]: Explore other loss functions", step)
            loss = model.calc_loss(output, target)
            # calculate deriviative of model.forward() and put it in model.parameters()...gradient
            loss.backward()
            # print progress of the learning
            # self.print_stats(step, loss, correct, total)
            # update model: model.parameters() -= lr * gradient
            optimizer.step()
            step += 1
        return step
    
    # def print_stats(self, step, loss, correct, total):
    #     if step % 500 == 0:
    #         print("Prediction = {}/{} Error = {} Learning rate = {} Hidden1 size = {} Hidden2 size = {} Step = {}".format(
    #             correct, total, loss.item(), self.learning_rate, self.hidden_size, self.hidden_size, step))

###
###
### Don't change code after this line
###
###
class Limits:
    def __init__(self):
        self.time_limit = 2.0
        self.size_limit = 10000
        self.test_limit = 1.0

class DataProvider:
    def __init__(self):
        self.number_of_cases = 10

    def create_data(self, input_size, seed):
        random.seed(seed)
        data_size = 1 << input_size
        data = torch.FloatTensor(data_size, input_size)
        target = torch.FloatTensor(data_size)
        for i in range(data_size):
            for j in range(input_size):
                input_bit = (i>>j)&1
                data[i,j] = float(input_bit)
            target[i] = float(random.randint(0, 1))
        return (data, target.view(-1, 1))

    def create_case_data(self, case):
        input_size = min(3+case, 7)
        data, target = self.create_data(input_size, case)
        return sm.CaseData(case, Limits(), (data, target), (data, target)).set_description("{} inputs".format(input_size))


class Config:
    def __init__(self):
        self.max_samples = 1000

    def get_data_provider(self):
        return DataProvider()

    def get_solution(self):
        return Solution()

# If you want to run specific case, put number here
sm.SolutionManager(Config()).run(case_number=-1)
