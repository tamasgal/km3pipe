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
            try { 
                steps {
                        sh """
                            . venv/bin/activate
                            make install-dev
                        """
                }
            } catch (e) { 
                rocketSend channel: '#km3pipe', message: "Build Failed - ${env.JOB_NAME} ${env.BUILD_NUMBER} (<${env.BUILD_URL}|Open>)"
                throw e
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
