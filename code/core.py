import torch
from scipy.stats import norm, binom_test
import numpy as np
from math import ceil
from statsmodels.stats.proportion import proportion_confint
from numerical import numerical_radius, calculate_radius


class Smooth(object):
    """A smoothed classifier g """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, base_classifier: torch.nn.Module, num_classes: int, sigma: float):
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param num_classes:
        :param sigma: the noise level hyperparameter
        """
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigma = sigma

    def predict(self, x: torch.tensor, n: int, alpha: float, batch_size: int) -> int:
        """ Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
        class returned by this method will equal g(x).

        This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.

        :param x: the input [channel x height x width]
        :param n: the number of Monte Carlo samples to use
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: the predicted class, or ABSTAIN
        """
        self.base_classifier.eval()
        counts = self._sample_noise(x, n, batch_size)
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            return Smooth.ABSTAIN
        else:
            return top2[0]

    def certify(self, x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int, label) -> (int, float):

        self.base_classifier.eval()
        # draw samples of f(x+ epsilon)
        counts_selection = self._sample_noise(x, n0, batch_size)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax().item()
        # draw more samples of f(x + epsilon)
        if cAHat != label:
            return 0, 0.0, 0.0, 0.0, 0.0, 0.0, cAHat, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        nA, maxC, meanC, meanR, meanG, meanB, n1, n2 = self._sample_noise_first(
            x, n, cAHat, batch_size)
        # use these samples to estimate a lower bound on pA
        pABar = self._lower_confidence_bound(nA, n, alpha/2)
        # use these samples to get statistical estimates on vaiours norms of gradient of pA wrt x
        maxDir, meanDir, meanRpar, meanRperp, meanGpar, meanGperp, meanBpar, meanBperp, meanBarL2, meanBarOp = self._confidence_bound(
            maxC, meanC, meanR, meanG, meanB, n1, n2, np.prod(x.shape), alpha/2)

        if pABar < 0.5:
            return nA, maxC, meanC, meanR, meanG, meanB, Smooth.ABSTAIN, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        else:
            radiusL2 = self.sigma * calculate_radius(pABar, meanBarL2)
            if abs(meanBarOp) < abs(maxDir) :
                radiusL1 = radiusL2 
            else :
                radiusL1 = self.sigma * numerical_radius(pABar, np.sqrt(meanBarOp**2 - maxDir**2), -maxDir)
            radiusR = self.sigma * numerical_radius(pABar, meanRperp, meanRpar)
            radiusG = self.sigma * numerical_radius(pABar, meanGperp, meanGpar)
            radiusB = self.sigma * numerical_radius(pABar, meanBperp, meanBpar)
            
            if meanDir > abs(meanBarOp) :
                radiusLInf = radiusL2
            else:
                radiusLInf = self.sigma * numerical_radius(pABar, np.sqrt(meanBarOp**2 - meanDir**2), -meanDir)
            radiusLInf /= np.sqrt(np.prod(x.shape))
            
            radiusOp = self.sigma * calculate_radius(pABar, meanBarOp)
            pABar = self._lower_confidence_bound(nA, n, 2*alpha)
            radiusCohen = self.sigma * norm.ppf(pABar)
            return nA, maxC, meanC, meanR, meanG, meanB, cAHat, radiusR, radiusG, radiusB, radiusL1, radiusLInf, radiusL2, radiusOp, radiusCohen

    def _sample_noise(self, x: torch.tensor, num: int, batch_size) -> np.ndarray:
        """ Sample the base classifier's prediction under noisy corruptions of the input x.

        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        with torch.no_grad():
            counts = np.zeros(self.num_classes, dtype=int)
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                batch = x.repeat((this_batch_size, 1, 1, 1))
                noise = torch.randn_like(batch, device='cuda') * self.sigma
                predictions = self.base_classifier(batch + noise).argmax(1)
                counts += self._count_arr(predictions.cpu().numpy(),
                                          self.num_classes)
            return counts

    def _sample_noise_first(self, x: torch.tensor, num: int, pred_class: int, batch_size) -> (int, np.ndarray, int, int):
        """ Sample the base classifier's prediction under noisy corruptions of the input x.

        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :param pred_class : the class expected to be predictied with the highest probability 
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        with torch.no_grad():
            counts = 0
            mean1 = torch.zeros_like(x)
            mean2 = torch.zeros_like(x)
            n1, n2 = 0, 0
            for i in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                batch = x.repeat((this_batch_size, 1, 1, 1))
                noise = torch.randn_like(batch, device='cuda') * self.sigma
                predictions = self.base_classifier(batch + noise).argmax(1)
                predictions = (predictions == pred_class).type(torch.double)

                counts += torch.sum(predictions).item()
                if i % 2 == 0:
                    mean1 += torch.einsum(
                        'nijk,n->ijk', [noise, (predictions - 0.5).type(torch.FloatTensor).to('cuda')])
                    n1 += this_batch_size
                else:
                    mean2 += torch.einsum(
                        'nijk,n->ijk', [noise, (predictions - 0.5).type(torch.FloatTensor).to('cuda')])
                    n2 += this_batch_size
            maxC = max(torch.max((mean1 + mean2)/(n1 + n2)).item(), -torch.min((mean1 + mean2)/(n1 + n2)).item())
            meanC = torch.abs((mean1 + mean2)/(n1 + n2)).sum().item()/np.sqrt(np.prod(x.shape))
            meanR, meanG, meanB = torch.mul(mean1/n1, mean2/n2).sum((1, 2)).cpu().data.numpy()/(self.sigma**2)
            return counts, maxC/self.sigma, meanC/self.sigma, meanR, meanG, meanB, n1, n2

    def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.

        This function uses the Clopper-Pearson method.

        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]

    def _confidence_bound(self, maxC : float, meanC : float, meanR: float, meanG: float, meanB: float, N1: int, N2: int, D: int, alpha: float) -> float:
        """ Returns a (1 - alpha) confidence bound on the norm values of the gradient

        These values are based on the derivations given in the paper.

        :param maxC: the empirical estimate of the infinity-norm of the gradient 
        :param meanC: the empirical estimate of the 1-norm of the gradient 
        :param meanR: the empirical estimate of the 2-norm of the gradient in red-channel
        :param meanG: the empirical estimate of the 2-norm of the gradient in green-channel
        :param meanB: the empirical estimate of the 2-norm of the gradient in blue-channel
        :param N1: the number of total draws used to calculate an independent empirical estimate of the gradient
        :param N1: the number of total draws used to calculate a second independent empirical estimate of the gradient
        :param D: the dimensionality of the input space
        :param alpha: the confidence level
        :return: the relevant bounds on the infinity-norm, 1-norm, color-space 2-norm, 2-norm
                of the gradient each of which holds true w.p at least (1 - alpha) over the samples
        """

        c = (0.25 + (3/np.sqrt(8*np.pi*np.e)))
        t0 = np.sqrt(2*(c/(N1 + N2))*np.log(4*D/alpha))
        maxDir = maxC + t0
        
        t0 = np.sqrt(2*c*(D*np.log(2) - np.log(alpha/2))/(N1 + N2))
        meanDir = meanC + t0
        
        t1 = c*np.sqrt(-2*D*np.log(alpha/2)/(N1*N2))

        meanA = meanR + meanG + meanB
        epsL2 = np.sqrt(-c*(N1 + N2)*np.log(alpha/2)/((meanA + t1)*2*N1*N2))
        meanL2 = np.sqrt(meanA + t1)/(np.sqrt(1 + epsL2**2) - epsL2)

        if ((meanA - t1) <= 0):
            meanOp = 0.0
        else:
            epsOp = np.sqrt(-c*(N1 + N2)*np.log(alpha/2)/((meanA - t1)*2*N1*N2))
            meanOp = -np.sqrt(meanA - t1)/(np.sqrt(1 + epsOp**2) + epsOp)

        t2 = c*np.sqrt(-2*D*np.log(alpha/2)/(3*N1*N2))

        eps = np.sqrt(-c*(N1 + N2)* np.log(alpha/2)/((meanR + t2)*2*N1*N2))
        meanRpar = -np.sqrt(meanR + t2)/(np.sqrt(1 + eps**2) - eps)
        if ((meanG + meanB - t2) <= 0):
            meanRperp = 0.0
        else:
            eps = np.sqrt(-c*(N1 + N2)*np.log(alpha/2)/((meanG + meanB - t2)*2*N1*N2))
            meanRperp = np.sqrt(meanG + meanB - t2)/(np.sqrt(1 + eps**2) + eps)

        eps = np.sqrt(-c*(N1 + N2) * np.log(alpha/2)/((meanB + t2)*2*N1*N2))
        meanBpar = -np.sqrt(meanB + t2)/(np.sqrt(1 + eps**2) - eps)
        if ((meanG + meanR - t2) <= 0):
            meanBperp = 0.0
        else:
            eps = np.sqrt(-c*(N1 + N2)*np.log(alpha/2) / ((meanG + meanR - t2)*2*N1*N2))
            meanBperp = np.sqrt(meanG + meanR - t2) / (np.sqrt(1 + eps**2) + eps)

        eps = np.sqrt(-c*(N1 + N2) * np.log(alpha/2)/((meanG + t2)*2*N1*N2))
        meanGpar = -np.sqrt(meanG + t2)/(np.sqrt(1 + eps**2) - eps)
        if ((meanR + meanB - t2) <= 0):
            meanGperp = 0.0
        else:
            eps = np.sqrt(-c*(N1 + N2)*np.log(alpha/2) / ((meanR + meanB - t2)*2*N1*N2))
            meanGperp = np.sqrt(meanR + meanB - t2) / (np.sqrt(1 + eps**2) + eps)

        return maxDir, meanDir, meanRpar, meanRperp, meanGpar, meanGperp, meanBpar, meanBperp, meanL2, meanOp
