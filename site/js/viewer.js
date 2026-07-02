/**
 * Three.js 3D Viewer for Face Visualization
 */

class FaceViewer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.model = null;
        
        this.init();
    }
    
    init() {
        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x1a1a2e);
        
        // Camera
        this.camera = new THREE.PerspectiveCamera(
            75,
            this.container.clientWidth / this.container.clientHeight,
            0.1,
            1000
        );
        this.camera.position.z = 2;
        
        // Renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(
            this.container.clientWidth,
            this.container.clientHeight
        );
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.container.appendChild(this.renderer.domElement);
        
        // Controls
        this.controls = new THREE.OrbitControls(
            this.camera,
            this.renderer.domElement
        );
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        
        // Lighting
        this.setupLighting();
        
        // Grid helper
        this.setupGrid();
        
        // Handle resize
        window.addEventListener('resize', () => this.onResize());
        
        // Start animation loop
        this.animate();
    }
    
    setupLighting() {
        // Ambient light
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        this.scene.add(ambientLight);
        
        // Directional light
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
        directionalLight.position.set(1, 1, 1);
        this.scene.add(directionalLight);
        
        // Back light
        const backLight = new THREE.DirectionalLight(0xffffff, 0.3);
        backLight.position.set(-1, -1, -1);
        this.scene.add(backLight);
    }
    
    setupGrid() {
        // Add grid for reference
        const gridHelper = new THREE.GridHelper(10, 10, 0x444444, 0x222222);
        gridHelper.rotation.x = Math.PI / 2;
        gridHelper.position.z = -1;
        this.scene.add(gridHelper);
    }
    
    loadModel(url) {
        // Remove existing model
        if (this.model) {
            this.scene.remove(this.model);
        }
        
        // Show loading indicator
        this.showLoading(true);
        
        const loader = new THREE.GLTFLoader();
        
        loader.load(
            url,
            (gltf) => {
                this.model = gltf.scene;
                
                // Center model
                const box = new THREE.Box3().setFromObject(this.model);
                const center = box.getCenter(new THREE.Vector3());
                this.model.position.sub(center);
                
                // Scale to fit
                const size = box.getSize(new THREE.Vector3());
                const maxDim = Math.max(size.x, size.y, size.z);
                const scale = 2 / maxDim;
                this.model.scale.multiplyScalar(scale);
                
                this.scene.add(this.model);
                this.showLoading(false);
                
                console.log('Model loaded successfully');
            },
            (progress) => {
                console.log('Loading:', (progress.loaded / progress.total * 100) + '%');
            },
            (error) => {
                console.error('Error loading model:', error);
                this.showLoading(false);
                this.showError('Ошибка загрузки модели');
            }
        );
    }
    
    loadFromGeometry(vertices, faces) {
        // Remove existing model
        if (this.model) {
            this.scene.remove(this.model);
        }
        
        // Create geometry
        const geometry = new THREE.BufferGeometry();
        
        // Set vertices
        const positionArray = new Float32Array(vertices.flat());
        geometry.setAttribute('position', new THREE.BufferAttribute(positionArray, 3));
        
        // Set faces
        const indexArray = new Uint32Array(faces.flat());
        geometry.setIndex(new THREE.BufferAttribute(indexArray, 1));
        
        // Compute normals
        geometry.computeVertexNormals();
        
        // Create material
        const material = new THREE.MeshStandardMaterial({
            color: 0xffd4b8,
            roughness: 0.7,
            metalness: 0.1,
            side: THREE.DoubleSide
        });
        
        // Create mesh
        this.model = new THREE.Mesh(geometry, material);
        
        // Center model
        geometry.computeBoundingBox();
        const center = geometry.boundingBox.getCenter(new THREE.Vector3());
        this.model.position.sub(center);
        
        // Scale to fit
        const size = geometry.boundingBox.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        const scale = 2 / maxDim;
        this.model.scale.multiplyScalar(scale);
        
        this.scene.add(this.model);
    }
    
    showLoading(show) {
        const placeholder = this.container.querySelector('.placeholder');
        if (placeholder) {
            placeholder.style.display = show ? 'block' : 'none';
        }
    }
    
    showError(message) {
        const placeholder = this.container.querySelector('.placeholder');
        if (placeholder) {
            placeholder.innerHTML = `<p style="color: #ef4444;">${message}</p>`;
        }
    }
    
    resetCamera() {
        this.camera.position.z = 2;
        this.controls.reset();
    }
    
    onResize() {
        this.camera.aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(
            this.container.clientWidth,
            this.container.clientHeight
        );
    }
    
    animate() {
        requestAnimationFrame(() => this.animate());
        
        // Rotate model slowly
        if (this.model) {
            this.model.rotation.y += 0.005;
        }
        
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }
    
    takeScreenshot() {
        this.renderer.render(this.scene, this.camera);
        return this.renderer.domElement.toDataURL('image/png');
    }
}

// Export for use in other modules
window.FaceViewer = FaceViewer;
