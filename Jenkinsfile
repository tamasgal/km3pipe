def docker_images = ["python:3.6.4"]

def get_stages(docker_image) {
    stages = {
        docker.image(docker_image).inside {
            stage("${docker_image}") {
                echo 'Running in ${docker_image}'
            }
            stage("Prepare") {
                if(docker_image == "python:2.7.14") {
                    sh """
                        virtualenv venv
                    """
                } else {
                    sh 'python -m venv venv'
                    sh """
                        . venv/bin/activate
                        pip install -U pip setuptools wheel
                    """
                }
            }
            stage("Build") {
                try { 
                    sh """
                        . venv/bin/activate
                        make
                    """
                } catch (e) { 
                    rocketSend channel: '#km3pipe', message: "Build Failed - ${env.JOB_NAME} ${env.BUILD_NUMBER} (<${env.BUILD_URL}|Open>)"
                    throw e
                }
            }
            stage("Install") {
                try { 
                    sh """
                        . venv/bin/activate
                        make install
                    """
                } catch (e) { 
                    rocketSend channel: '#km3pipe', message: "Install Failed - ${env.JOB_NAME} ${env.BUILD_NUMBER} (<${env.BUILD_URL}|Open>)"
                    throw e
                }
            }
            stage('Test') {
                try { 
                    sh """
                        . venv/bin/activate
                        make clean
                        make test
                    """
                    junit 'junit.xml'
                } catch (e) { 
                    rocketSend channel: '#km3pipe', message: "Test Suite Failed - ${env.JOB_NAME} ${env.BUILD_NUMBER} (<${env.BUILD_URL}|Open>)"
                    throw e
                }
            }
            stage('Docs') {
                try { 
                    sh """
                        . venv/bin/activate
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
