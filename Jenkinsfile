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
                    source venv/bin/activate
                    make install-dev
                """
            }
        }
        stage('Test') {
            steps {
                sh """
                    source venv/bin/activate
                    make test
                """
            }
        }
    }
}
