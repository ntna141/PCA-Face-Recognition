'''pca_cov.py
Performs principal component analysis using the covariance matrix of the dataset
YOUR NAME HERE
CS 251: Data Analysis and Visualization
Fall 2023
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from data_transformations import normalize, center


class PCA:
    '''Perform and store principal component analysis results

    NOTE: In your implementations, only the following "high level" `scipy`/`numpy` functions can be used:
    - `np.linalg.eig`
    The numpy functions that you have been using so far are fine to use.
    '''

    def __init__(self, data):
        '''

        Parameters:
        -----------
        data: pandas DataFrame. shape=(num_samps, num_vars)
            Contains all the data samples and variables in a dataset. Should be set as an instance variable.
        '''
        self.data = data

        # vars: Python list. len(vars) = num_selected_vars
        #   String variable names selected from the DataFrame to run PCA on.
        #   num_selected_vars <= num_vars
        self.vars = None

        # A: ndarray. shape=(num_samps, num_selected_vars)
        #   Matrix of data selected for PCA
        self.A = None

        # normalized: boolean.
        #   Whether data matrix (A) is normalized by self.pca
        self.normalized = None

        # A_proj: ndarray. shape=(num_samps, num_pcs_to_keep)
        #   Matrix of PCA projected data
        self.A_proj = None

        # e_vals: ndarray. shape=(num_pcs,)
        #   Full set of eigenvalues (ordered large-to-small)
        self.e_vals = None
        # e_vecs: ndarray. shape=(num_selected_vars, num_pcs)
        #   Full set of eigenvectors, corresponding to eigenvalues ordered large-to-small
        self.e_vecs = None

        # prop_var: Python list. len(prop_var) = num_pcs
        #   Proportion variance accounted for by the PCs (ordered large-to-small)
        self.prop_var = None

        # cum_var: Python list. len(cum_var) = num_pcs
        #   Cumulative proportion variance accounted for by the PCs (ordered large-to-small)
        self.cum_var = None

        # orig_means: ndarray. shape=(num_selected_vars,)
        #   Means of each orignal data variable
        self.orig_means = None

        # orig_mins: ndarray. shape=(num_selected_vars,)
        #   Mins of each orignal data variable
        self.orig_mins = None

        # orig_maxs: ndarray. shape=(num_selected_vars,)
        #   Maxs of each orignal data variable
        self.orig_maxs = None

    def get_prop_var(self):
        '''(No changes should be needed)'''
        return self.prop_var

    def get_cum_var(self):
        '''(No changes should be needed)'''
        return self.cum_var

    def get_eigenvalues(self):
        '''(No changes should be needed)'''
        return self.e_vals

    def get_eigenvectors(self):
        '''(No changes should be needed)'''
        return self.e_vecs

    def covariance_matrix(self, data):
        '''Computes the covariance matrix of `data`

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_vars)
            `data` is NOT centered coming in, you should do that here.

        Returns:
        -----------
        ndarray. shape=(num_vars, num_vars)
            The covariance matrix of centered `data`

        NOTE: You should do this wihout any loops
        NOTE: np.cov is off-limits here — compute it from "scratch"!
        '''
        if not np.allclose(np.mean(data, axis=0), 0):
            data = center(data)  

        cov_matrix = np.dot(data.T, data) / (data.shape[0] - 1)

        return cov_matrix

    def compute_prop_var(self, e_vals):
        '''Computes the proportion variance accounted for by the principal components (PCs).

        Parameters:
        -----------
        e_vals: ndarray. shape=(num_pcs,)

        Returns:
        -----------
        Python list. len = num_pcs
            Proportion variance accounted for by the PCs
        '''
        total_variance = np.sum(e_vals) 
        prop_var = [eigval / total_variance for eigval in e_vals] 

        return prop_var

    def compute_cum_var(self, prop_var):
        '''Computes the cumulative variance accounted for by the principal components (PCs).

        Parameters:
        -----------
        prop_var: Python list. len(prop_var) = num_pcs
            Proportion variance accounted for by the PCs, ordered largest-to-smallest
            [Output of self.compute_prop_var()]

        Returns:
        -----------
        Python list. len = num_pcs
            Cumulative variance accounted for by the PCs
        '''
        cum_var = [sum(prop_var[:i+1]) for i in range(len(prop_var))]
        return cum_var

    def pca(self, vars, normalize_dataset=False):
        '''Performs PCA on the data variables `vars`

        Parameters:
        -----------
        vars: Python list of strings. len(vars) = num_selected_vars
            1+ variable names selected to perform PCA on.
            Variable names must match those used in the `self.data` DataFrame.
        normalize_dataset: boolean.
            If True, min-max normalize each data variable it ranges from 0 to 1.

        NOTE: Leverage other methods in this class as much as possible to do computations.

        TODO:
        - Select the relevant data (corresponding to `vars`) from the data pandas DataFrame
        then convert to numpy ndarray for forthcoming calculations.
        - If `normalize` is True, normalize the selected data so that each variable (column)
        ranges from 0 to 1 (i.e. normalize based on the dynamic range of each variable).
            - Before normalizing, create instance variables containing information that would be
            needed to "undo" or reverse the normalization on the selected data.
        - Make sure to compute everything needed to set all instance variables defined in constructor,
        except for self.A_proj (this will happen later).
        '''
        self.vars = vars
        if vars is not None:
            self.A = self.data[vars].to_numpy()
        else:
            self.A = self.data


        if normalize_dataset:
            self.orig_mins = self.A.min(axis=0)
            self.orig_maxs = self.A.max(axis=0)
            self.A = normalize(self.A)
            

        Ac= center(self.A)

        cov_matrix = self.covariance_matrix(Ac)

        e_vals, e_vecs = np.linalg.eig(cov_matrix)
        sorted_indices = np.argsort(e_vals)[::-1]
        e_vals = e_vals[sorted_indices]
        e_vecs = e_vecs[:, sorted_indices]

        self.e_vals = e_vals
        self.e_vecs = e_vecs
        self.prop_var = self.compute_prop_var(e_vals)
        self.cum_var = self.compute_cum_var(self.prop_var)

        self.normalized = normalize_dataset

        

    def elbow_plot(self, num_pcs_to_keep=None):
        '''Plots a curve of the cumulative variance accounted for by the top `num_pcs_to_keep` PCs.
        x axis corresponds to top PCs included (large-to-small order)
        y axis corresponds to proportion variance accounted for

        Parameters:
        -----------
        num_pcs_to_keep: int. Show the variance accounted for by this many top PCs.
            If num_pcs_to_keep is None, show variance accounted for by ALL the PCs (the default).

        NOTE: Make plot markers at each point. Enlarge them so that they look obvious.
        NOTE: Reminder to create useful x and y axis labels.
        NOTE: Don't write plt.show() in this method
        '''
        if self.cum_var is None:
            raise ValueError('Cant plot cumulative variance. Compute the PCA first.')
        if num_pcs_to_keep is None:
            num_pcs_to_keep = len(self.cum_var)  

        x = list(range(1, num_pcs_to_keep + 1))
        y = self.cum_var[:num_pcs_to_keep]

        plt.figure(figsize=(10, 10)) 
        plt.plot(x, y, marker='o', linestyle='-', color='b', markersize=8)
        # plt.xlim(160, 220)

    def pca_project(self, pcs_to_keep):
        '''Project the data onto `pcs_to_keep` PCs (not necessarily contiguous)

        Parameters:
        -----------
        pcs_to_keep: Python list of ints. len(pcs_to_keep) = num_pcs_to_keep
            Project the data onto these PCs.
            NOTE: This LIST contains indices of PCs to project the data onto, they are NOT necessarily
            contiguous.
            Example 1: [0, 2] would mean project on the 1st and 3rd largest PCs.
            Example 2: [0, 1] would mean project on the two largest PCs.

        Returns
        -----------\
        pca_proj: ndarray. shape=(num_samps, num_pcs_to_keep).
            e.g. if pcs_to_keep = [0, 1],
            then pca_proj[:, 0] are x values, pca_proj[:, 1] are y values.

        NOTE: This method should set the variable `self.A_proj`
        '''
        selected_e_vecs = self.e_vecs[:, pcs_to_keep]

        pca_proj = np.dot(self.data, selected_e_vecs)


        self.A_proj = pca_proj

        return pca_proj

    def pca_then_project_back(self, top_k):
        '''Project the data into PCA space (on `top_k` PCs) then project it back to the data space

        (Week 2)

        Parameters:
        -----------
        top_k: int. Project the data onto this many top PCs.

        Returns:
        -----------
        ndarray. shape=(num_samps, num_selected_vars)

        TODO:
        - Project the data on the `top_k` PCs (assume PCA has already been performed).
        - Project this PCA-transformed data back to the original data space
        - If you normalized, remember to rescale the data projected back to the original data space.
        '''
        if top_k <= 0:
            raise ValueError('top_k should be a positive integer.')

        if top_k > self.e_vecs.shape[1]:
            raise ValueError('top_k cannot be greater than the number of available PCs.')

        selected_e_vecs = self.e_vecs[:, :top_k]
        # print(self.e_vecs[:, :top_k])

        pca_proj = self.pca_project(np.arange(top_k))
        if self.normalized:
            self.A = self.A * (self.orig_maxs - self.orig_mins) + self.orig_mins
        self.A_proj = pca_proj
        self.A = np.dot(self.A_proj, selected_e_vecs.T)  

        

        return self.A

    def loading_plot(self):
        '''Create a loading plot of the top 2 PC eigenvectors

        (Week 2)

        TODO:
        - Plot a line joining the origin (0, 0) and corresponding components of the top 2 PC eigenvectors.
            Example: If e_0 = [0.1, 0.3] and e_1 = [1.0, 2.0], you would create two lines to join
            (0, 0) and (0.1, 1.0); (0, 0) and (0.3, 2.0).
            Number of lines = num_vars
        - Use plt.annotate to label each line by the variable that it corresponds to.
        - Reminder to create useful x and y axis labels.
        '''
        if self.e_vecs is None:
            raise ValueError('Eigenvectors have not been computed. Please run PCA first.')

        top_2_eigenvectors = self.e_vecs[:, :2]
        # print(top_2_eigenvectors.shape)
    
        plt.figure(figsize=(8, 6))

        for i in range(top_2_eigenvectors.shape[0]):
            plt.plot([0, top_2_eigenvectors[i, 0]], [0, top_2_eigenvectors[i, 1]], marker='o', label=f'{list(self.data.columns)[i]}')
            # print(top_2_eigenvectors[i,0], top_2_eigenvectors[i,1])
            plt.annotate(f'{list(self.data.columns)[i]}', (top_2_eigenvectors[i, 0], top_2_eigenvectors[i, 1]))

        plt.xlabel('PC1 Loading')
        plt.ylabel('PC2 Loading')
        plt.title('Loading Plot of Top 2 PC Eigenvectors')

        plt.legend()

        plt.grid()
        plt.show()


