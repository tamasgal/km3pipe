pipeline {
    agent {
        docker { image 'python:3.6.5' }
    }
    stages {
        stage('Prepare') {
            steps {
                sh 'python -m venv venv'
            }
        }
        stage('Build') {
            steps {
                sh """
                    . venv/bin/activate
                    make install-dev
                """
            }
        }
        stage('Test') {
            steps {
                sh """
                    . venv/bin/activate
                    make test
                """
            }
        }
        stage('Docs') {
            steps {
                sh """
                    . venv/bin/activate
                    make doc-dependencies
                    cd docs
                    make html
                """
            }
        }
    }
}