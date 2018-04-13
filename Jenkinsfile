def docker_images = ["python:2.7.14", "python:3.5.4", "python:3.6.2"]

def get_stages(docker_image) {
    stages = {
        docker.image(docker_image).inside {
            stage("${docker_image}") {
                echo 'Running in ${docker_image}'
            }
            stage("Prepare") {
                switch (docker_image) {
                    case "python:2.7.14":
                        sh 'exit 1'
                        break
                    default:
                        sh 'python -m venv venv'
                }
            }
            stage("Build") {
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
                        export MPL_BACKEND="agg"
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
    def stages = [:]

    for (int i = 0; i < docker_images.size(); i++) {
        def docker_image = docker_images[i]
        stages[docker_image] = get_stages(docker_image)
    }

    checkout scm
    parallel stages
}
