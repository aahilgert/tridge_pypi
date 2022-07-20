# tridge_pypi

The Tridge package allows tuning free ridge regression under gaussian, binomial, and poisson distributions.

### GaussianRegressor

#### Parameters:

##### nlambda : int, default=100

Number of candidate lambdas to be generated.

##### lambda_min : float, default=0.05

Minimum value of candidate lambda.

##### c : int, default=5

Half of range of candidate lambdas to be generated.

##### processor : {'cpu','gpu'}, default='cpu'

Processor to be used. If the `r_type` is incompatable with the processor type, the regressor will generate as default.

##### r_type : {'woodbury','svd','rsvd','sklearn','custom'}, default='woodbury'

Technique to be used. `woodbury`, `svd`, and `rsvd` are compatible with both the cpu and gpu. `sklearn` is only compatible with the cpu. `Custom` requires that a custom function is provided.

##### function : float -> ndarray of shape (n_features,), default=None

Custom function to generate estimators from candidate lambdas. If `Custom` is chosen `r_type` and no function is provided, the regressor will generate as default.

#### Attributes:

##### lambdas : ndarray of shape (nlambda,)

Generated candidate lambdas.

##### edr_lambda : float

Median candidate lambda.

##### best_lambda : float

Lambda of best objective function cost.

##### best_cost : float

Cost of chosen candidate lambda. 

##### coef_ : ndarray of shape (n_features,)

Coefficients of selected estimator.

#### Methods:

##### gen_lambdas(X : ndarray of shape (n_samples,n_features), y : ndarray of shape (n_samples,))

```
from tridge import GaussianRegressor
>>> clf = GaussianRegressor()
>>> clf.gen_lambdas(X, y)
>>> clf.lambdas # candidate lambdas
```

##### fit(X : ndarray of shape (n_samples,n_features), y : ndarray of shape (n_samples,))

```
from tridge import GaussianRegressor
>>> clf = GaussianRegressor()
>>> clf.fit(X, y)
>>> clf.coef_ # will return estimator
>>> clf.lambdas # candidate lambdas
>>> clf.best_cost # cost of chosen candidate lambda
>>> clf.best_lambda # candidate lambda of best cost
```

##### predict(X : ndarray of shape (float,n_features))

If the model has not been already fit, returns value error.

```
from tridge import GaussianRegressor
>>> clf = GaussianRegressor()
>>> clf.fit(X, y)
>>> clf.predict(X_pred)
# returns prediction
```


### BinomialRegressor

#### Parameters:

##### nlambda : int, default=100

Number of candidate lambdas to be generated.

##### lambda_min : float, default=0.05

Minimum value of candidate lambda.

##### c : int, default=5

Half of range of candidate lambdas to be generated.

##### processor : {'cpu','gpu'}, default='cpu'

Processor to be used. If the `r_type` is incompatible with the processor type, the regressor will generate as default.

##### r_type : {'sklearn','rls','custom'}, default='sklearn'

Technique to be used. `sklearn` is only compatible with the cpu. Reweighted least squares (`rls`) is compatible with both the cpu and gpu. `Custom` requires that a custom function is provided.

##### function : float -> ndarray of shape (n_features,), default=None

Custom function to generate estimators from candidate lambdas. If `Custom` is chosen `r_type` and no function is provided, the regressor will generate as default.

#### Attributes:

##### lambdas : ndarray of shape (nlambda,)

Generated candidate lambdas.

##### edr_lambda : float

Median candidate lambda.

##### best_lambda : float

Lambda of best objective function cost.

##### best_cost : float

Cost of chosen candidate lambda. 

##### coef_ : ndarray of shape (n_features,)

Coefficients of selected estimator.

#### Methods:

##### gen_lambdas(X : ndarray of shape (n_samples,n_features), y : ndarray of shape (n_samples,))

```
from tridge import BinomialRegressor
>>> clf = BinomialRegressor()
>>> clf.gen_lambdas(X, y)
>>> clf.lambdas # candidate lambdas
```

##### fit(X : ndarray of shape (n_samples,n_features), y : ndarray of shape (n_samples,))

```
from tridge import BinomialRegressor
>>> clf = BinomialRegressor()
>>> clf.fit(X, y)
>>> clf.coef_ # will return estimator
>>> clf.lambdas # candidate lambdas
>>> clf.best_cost # cost of chosen candidate lambda
>>> clf.best_lambda # candidate lambda of best cost
```

##### predict(X : ndarray of shape (float,n_features))

If the model has not been already fit, returns value error.

```
from tridge import BinomialRegressor
>>> clf = BinomialRegressor()
>>> clf.fit(X, y)
>>> clf.predict(X_pred)
# returns prediction
```


### PoissonRegressor

#### Parameters:

##### nlambda : int, default=100

Number of candidate lambdas to be generated.

##### lambda_min : float, default=0.05

Minimum value of candidate lambda.

##### c : int, default=5

Half of range of candidate lambdas to be generated.

##### processor : {'cpu','gpu'}, default='cpu'

Processor to be used.

##### r_type : {'rls','custom'}, default='rls'

Technique to be used. Reweighted least squares (`rls`) is compatible with both the cpu and gpu. `Custom` requires that a custom function is provided. 

##### function : float -> ndarray of shape (n_features,), default=None

Custom function to generate estimators from candidate lambdas. If `Custom` is chosen `r_type` and no function is provided, the regressor will generate as default.

#### Attributes:

##### lambdas : ndarray of shape (nlambda,)

Generated candidate lambdas.

##### edr_lambda : float

Median candidate lambda.

##### best_lambda : float

Lambda of best objective function cost.

##### best_cost : float

Cost of chosen candidate lambda. 

##### coef_ : ndarray of shape (n_features,)

Coefficients of selected estimator.

#### Methods:

##### gen_lambdas(X : ndarray of shape (n_samples,n_features), y : ndarray of shape (n_samples,))

```
from tridge import PoissonRegressor
>>> clf = PoissonRegressor()
>>> clf.gen_lambdas(X, y)
>>> clf.lambdas # candidate lambdas
```

##### fit(X : ndarray of shape (n_samples,n_features), y : ndarray of shape (n_samples,))

```
from tridge import PoissonRegressor
>>> clf = PoissonRegressor()
>>> clf.fit(X, y)
>>> clf.coef_ # will return estimator
>>> clf.lambdas # candidate lambdas
>>> clf.best_cost # cost of chosen candidate lambda
>>> clf.best_lambda # candidate lambda of best cost
```

##### predict(X : ndarray of shape (float,n_features))

If the model has not been already fit, returns value error.

```
from tridge import PoissonRegressor
>>> clf = PoissonRegressor()
>>> clf.fit(X, y)
>>> clf.predict(X_pred)
# returns prediction
```
