import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=False)

    # *** START CODE HERE ***
    modelGDA = GDA()
    modelGDA.fit(x_train,y_train)
    pred = modelGDA.predict(x_train)
    print("Accuracy on training set: ",np.mean(pred == y_train))
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        m,n = len(x),len(x[0])
        y_sum = sum(y)
        
        #phi: probability that label is 1
        phi = y_sum/m
        
        #mu0,mu1: expected feature variable given label 0/1
        mu_0 = (x.T @ (1-y))/(m-y_sum)
        mu_1 = (x.T @ y)/y_sum
        
        #sigma IKKE VEKTORISERET
        sigma = [[0.0,0.0],[0.0,0.0]]
        for i in range(m):
            if(y[i] == 0):
                sigma += np.outer(x[i]-mu_0,x[i]-mu_0)
            else:
                sigma += np.outer(x[i]-mu_1,x[i]-mu_1)
        sigma /= m
        
        #theta
        sigma_inv = np.linalg.inv(sigma)
        theta = np.dot(sigma_inv, mu_1 - mu_0)
        theta_0 = 1 / 2 * mu_0 @ sigma_inv @ mu_0 - 1 / 2 * mu_1 @ sigma_inv @ mu_1 - np.log((1 - phi) / phi)
        self.theta = np.insert(theta, 0, theta_0)
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        pred = 1/(1+np.exp(-(x@self.theta[1:]+self.theta[0])))
        return pred>=1/2
        # *** END CODE HERE