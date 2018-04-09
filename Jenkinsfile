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
                script {
                    try { 
                        sh """
                            . venv/bin/activate
                            make install-dev
                        """
                    } catch (e) { 
                        rocketSend channel: '#km3pipe', message: "Build Failed - ${env.JOB_NAME} ${env.BUILD_NUMBER} (<${env.BUILD_URL}|Open>)"
                        throw e
                    }
                }
            }
        }
        stage('Test') {
            steps {
                script {
                    try { 
                        sh """
                            . venv/bin/activate
                            make test
                        """
                    } catch (e) { 
                        rocketSend channel: '#km3pipe', message: "Test Suite Failed - ${env.JOB_NAME} ${env.BUILD_NUMBER} (<${env.BUILD_URL}|Open>)"
                        throw e
                    }
                }
            }
        }
        stage('Docs') {
            steps {
                script {
                    try { 
                        sh """
                            . venv/bin/activate
                            make doc-dependencies
                            cd docs
                            make html
                        """
                    } catch (e) { 
                        rocketSend channel: '#km3pipe', message: "Building the Docs Failed - ${env.JOB_NAME} ${env.BUILD_NUMBER} (<${env.BUILD_URL}|Open>)"
                        throw e
                    }
                }
            }
        }
    }
}
