def docker_images = ["python:3.5.5", "python:3.6.4", "python:3.6.5"]

def get_stages(docker_image) {
    stages = {
        docker.image(docker_image).inside {
            def PYTHON_VENV = "${docker_image + '_venv'}"

            stage("${docker_image}") {
                echo 'Running in ${docker_image}'
            }
            stage("Prepare") {
                sh 'rm -rf venv'
                sh 'python -m venv ${PYTHON_VENV}'
                sh """
                    . ${PYTHON_VENV}/bin/activate
                    pip install -U pip setuptools wheel
                """
            }
            stage("Build") {
                try { 
                    sh """
                        . ${PYTHON_VENV}/bin/activate
                        make
                    """
                } catch (e) { 
                    rocketSend channel: '#km3pipe', message: "Build Failed - ${env.JOB_NAME} ${env.BUILD_NUMBER} (<${env.BUILD_URL}|Open>)"
                    throw e
                }
            }
            stage("Install Dependencies") {
                try { 
                    sh """
                        . ${PYTHON_VENV}/bin/activate
                        make dependencies
                    """
                } catch (e) { 
                    rocketSend channel: '#km3pipe', message: "Install Dependencies Failed - ${env.JOB_NAME} ${env.BUILD_NUMBER} (<${env.BUILD_URL}|Open>)"
                    throw e
                }
            }
            stage("Install Doc Dependencies") {
                try { 
                    sh """
                        . ${PYTHON_VENV}/bin/activate
                        make doc-dependencies
                    """
                } catch (e) { 
                    rocketSend channel: '#km3pipe', message: "Install Doc Dependencies Failed - ${env.JOB_NAME} ${env.BUILD_NUMBER} (<${env.BUILD_URL}|Open>)"
                    throw e
                }
            }
            stage("Install Dev Dependencies") {
                try { 
                    sh """
                        . ${PYTHON_VENV}/bin/activate
                        make dev-dependencies
                    """
                } catch (e) { 
                    rocketSend channel: '#km3pipe', message: "Install Dev Dependencies Failed - ${env.JOB_NAME} ${env.BUILD_NUMBER} (<${env.BUILD_URL}|Open>)"
                    throw e
                }
            }
            stage('Test') {
                try { 
                    sh """
                        . ${PYTHON_VENV}/bin/activate
                        make clean
                        make test
                    """
                    junit 'junit.xml'
                } catch (e) { 
                    rocketSend channel: '#km3pipe', message: "Test Suite Failed - ${env.JOB_NAME} ${env.BUILD_NUMBER} (<${env.BUILD_URL}|Open>)"
                    throw e
                }
            }
            stage("Install") {
                try { 
                    sh """
                        . ${PYTHON_VENV}/bin/activate
                        make install
                    """
                } catch (e) { 
                    rocketSend channel: '#km3pipe', message: "Install Failed - ${env.JOB_NAME} ${env.BUILD_NUMBER} (<${env.BUILD_URL}|Open>)"
                    throw e
                }
            }
            stage('Test KM3Modules') {
                try { 
                    sh """
                        . ${PYTHON_VENV}/bin/activate
                        make test-km3modules
                    """
                    junit 'junit.xml'
                } catch (e) { 
                    rocketSend channel: '#km3pipe', message: "KM3Modules Test Suite Failed - ${env.JOB_NAME} ${env.BUILD_NUMBER} (<${env.BUILD_URL}|Open>)"
                    throw e
                }
            }
            stage('Docs') {
                try { 
                    sh """
                        . ${PYTHON_VENV}/bin/activate
                        make doc-dependencies
                        cd docs
                        export MPLBACKEND="agg"
                        make html
                    """
                } catch (e) { 
                    rocketSend channel: '#km3pipe', message: "Building the Docs Failed - ${env.JOB_NAME} ${env.BUILD_NUMBER} (<${env.BUILD_URL}|Open>)"
                    throw e
                }
            }

        }
    }
    return stages
}

node('master') {
    checkout scm

    def stages = [:]
    for (int i = 0; i < docker_images.size(); i++) {
        def docker_image = docker_images[i]
        stages[docker_image] = get_stages(docker_image)
        /* get_stages(docker_image) */
    }

    parallel stages

}
