import pandas as pd

class StandardScaler():
	def __init__(self):
		self.features = []
		self.mean = {}
		self.std = {}

	def fit(self, df: pd.DataFrame):
		self.features = df.select_dtypes(include='number').columns
		for f in self.features:
			lst = df[f]
			self.mean[f] = sum(lst) / len(lst)
			self.std[f] = (sum(abs(lst - self.mean[f]) ** 2) / (len(lst) - 1)) ** 0.5

	def transform(self, df: pd.DataFrame) -> pd.DataFrame:
		for f in self.features:
			df.loc[:,f] = (df.loc[:,f] - self.mean[f]) / self.std[f]
		return df

	def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
		self.fit(df)
		return self.transform(df)
	
	def save_to_file(self):
		try:
			scaler = pd.DataFrame()
			scaler['Features'] = pd.Series(self.features)
			scaler['Mean'] = pd.Series([v for k, v in self.mean.items()])
			scaler['Std'] = pd.Series([v for k, v in self.std.items()])
			scaler.to_csv('scaler.csv', index=False)
		except Exception as e:
			print(f'{type(e).__name__}: {e}')


