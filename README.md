# tridge_pypi

The Tridge package allows for examination of 

### GaussianRegressor

#### Parameters:

##### nlambda : int, default=100

##### lambda_min : float, default=0.05 

##### c : int, default=5 

##### processor : {'cpu','gpu'}, default='cpu'

##### r_type : {'woodbury','svd','rsvd','sklearn','custom'}, default='woodbury'

##### function : float -> ndarray of shape (n_features,), default=None

#### Attributes:

##### lambdas : ndarray of shape (nlambda,)

##### edr_lambda : float

##### best_lambda : float

##### best_cost : float

##### coef_ : ndarray of shape (n_features,)

#### Methods:

##### gen_lambdas(X : ndarray of shape (n_samples,n_features), y : ndarray of shape (n_samples,))



##### fit(X : ndarray of shape (n_samples,n_features), y : ndarray of shape (n_samples,))


### BinomialRegressor

#### Parameters:

##### nlambda : int, default=100

##### lambda_min : float, default=0.05 

##### c : int, default=5 

##### processor : {'cpu','gpu'}, default='cpu'

##### r_type : {'sklearn','rls','custom'}, default='sklearn'

##### function : float -> ndarray of shape (n_features,), default=None

#### Attributes:

##### lambdas : ndarray of shape (nlambda,)

##### edr_lambda : float

##### best_lambda : float

##### best_cost : float

##### coef_ : ndarray of shape (n_features,)

#### Methods:

##### gen_lambdas(X : ndarray of shape (n_samples,n_features), y : ndarray of shape (n_samples,))



##### fit(X : ndarray of shape (n_samples,n_features), y : ndarray of shape (n_samples,))


### PoissonRegressor

#### Parameters:

##### nlambda : int, default=100

##### lambda_min : float, default=0.05 

##### c : int, default=5 

##### processor : {'cpu','gpu'}, default='cpu'

##### r_type : {'rls','custom'}, default='rls'

##### function : float -> ndarray of shape (n_features,), default=None

#### Attributes:

##### lambdas : ndarray of shape (nlambda,)

##### edr_lambda : float

##### best_lambda : float

##### best_cost : float

##### coef_ : ndarray of shape (n_features,)

#### Methods:

##### gen_lambdas(X : ndarray of shape (n_samples,n_features), y : ndarray of shape (n_samples,))



##### fit(X : ndarray of shape (n_samples,n_features), y : ndarray of shape (n_samples,))
