import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(eval_path,add_intercept=True)

    # *** START CODE HERE ***
    
    #Fit model and plot decision boundary
    model = LogisticRegression()
    model.fit(x_train,y_train)
    print("The value of theta: ",model.theta)
    print("Plot of training data and decision boundary:")
    util.plot(x_train,y_train,model.theta,save_path = "plots/PS1-1-b")
    
    #accuracy training,evaluation
    print("Accuracy on training data: ",np.mean(y_train == model.predict(x_train)))
    print("Accuracy on evaluation data: ",np.mean(y_eval == model.predict(x_eval)))
    
    
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m,n = len(x),len(x[0])
        self.theta = np.zeros(n)
        while(True):
            X_theta = x @ self.theta
            sigmoid_X_theta = 1/(1+np.exp(-X_theta))
            
            #First derivative
            derivative_loss = -1/m * x.T @ (y-sigmoid_X_theta)
            
            #Hessian
            H = 1/m * x.T @ np.diag(sigmoid_X_theta) @ (np.identity(m)-np.diag(sigmoid_X_theta)) @ x
            
            #Invert Hessian
            H_inv = np.linalg.inv(H)
            
            #Compute new theta
            theta_new = self.theta - (H_inv @ derivative_loss)
            
            #Check if ||theta_old-theta_new||<=eps, and if so break
            if(np.linalg.norm(self.theta-theta_new) <= 1/10**5):
                break
            else:
                self.theta = theta_new
        
        #return self.theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        pred = 1/(1+np.exp(-x @ self.theta))
        return pred
        # *** END CODE HERE ***
