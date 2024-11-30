class StackingPipeline:
    def __init__(self, random_state=22):


        self.rf = RandomForestClassifier(random_state=random_state, class_weight='balanced')
        self.svc = SVC(random_state=random_state, class_weight='balanced', probability=True)
        self.lr = LogisticRegression(random_state=random_state, class_weight='balanced')
        
        self.preprocessor = Pipeline([
            ('robust_scaler', RobustScaler()),
            ('standard_scaler', StandardScaler())
        ])
        
        self.final_estimator = LogisticRegression(random_state=random_state, class_weight='balanced')

        self.model = StackingClassifier(
            estimators=[('rf', self.rf), ('svc', self.svc), ('lr', self.lr)],
            final_estimator=self.final_estimator
        )

        self.pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('stacking', self.model)
        ])

    def fit(self, X_train, y_train):
        """
        Обучение модели.
        """
        self.pipeline.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Предсказание на основе обученной модели.
        """
        return self.pipeline.predict(X_test)

    def evaluate(self, X_test, y_test):
        """
        Оценка качества модели с выводом отчета.
        """
        y_pred = self.predict(X_test)
        print(classification_report(y_test, y_pred))