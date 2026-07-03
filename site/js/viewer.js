import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const MODELS = {
    deca: {
        title: "DECA three-view coarse",
        description:
            "Экспериментальная сборка из трех проекций: общий identity shape усреднен по фронту и двум профилям, global pose обнулен для стабильного просмотра, текстура взята с фронтального кадра.",
        url: "models/deca_result.glb",
        vertices: "5023",
        faces: "9975",
        format: "GLB / TextureVisuals",
    },
    "deca-detail": {
        title: "DECA three-view detail",
        description:
            "High-detail экспорт через официальный DECA displacement upsample. Цветовая маска отключена, чтобы оценивать только плотную геометрию и микродетали без артефактов UV-проекции.",
        url: "models/deca_detail_neutral.glb",
        vertices: "59315",
        faces: "117380",
        format: "GLB / ColorVisuals",
    },
    "flame-textured": {
        title: "FLAME2023 textured baseline",
        description:
            "Параметрическая FLAME2023 Open модель с UV-разверткой и mean texture из FLAME_texture.npz. Это web-ready baseline для проверки текстурного экспорта.",
        url: "models/flame2023_textured.glb",
        vertices: "5023",
        faces: "9976",
        format: "GLB / TextureVisuals",
    },
    "flame-mask-baseline": {
        title: "FLAME2023 mask baseline",
        description:
            "Та же FLAME2023 Open сетка на 5k вершин, но mean texture запечена в vertex colors. Это стабильный baseline для просмотра FLAME texture-space маски без UV/PBR артефактов.",
        url: "models/flame2023_mask_baseline.glb",
        vertices: "5023",
        faces: "9976",
        format: "GLB / VertexColors",
    },
    "flame-neutral": {
        title: "FLAME2023 neutral mesh",
        description:
            "Нейтральная параметрическая голова без текстуры. Удобна для проверки топологии, масштаба и поведения материала в браузерной сцене.",
        url: "models/flame2023_neutral.glb",
        vertices: "5023",
        faces: "9976",
        format: "GLB / ColorVisuals",
    },
    "flame-detail-textured": {
        title: "FLAME2023 textured detail",
        description:
            "Уплотненный FLAME2023 Open baseline с сохраненной UV-разверткой и mean texture. Это subdivision-детализация без нейросетевого displacement-слоя.",
        url: "models/flame2023_detail_textured.glb",
        vertices: "59856",
        faces: "39904",
        format: "GLB / TextureVisuals",
    },
    "flame-detail-neutral": {
        title: "FLAME2023 detail mesh",
        description:
            "Та же уплотненная FLAME2023 Open геометрия без текстуры. Нужна для сравнения coarse/detail топологии FLAME отдельно от материала.",
        url: "models/flame2023_detail_neutral.glb",
        vertices: "59856",
        faces: "39904",
        format: "GLB / ColorVisuals",
    },
};

class FaceDemoViewer {
    constructor(container) {
        this.container = container;
        this.status = document.getElementById("viewer-status");
        this.download = document.getElementById("viewer-download");
        this.title = document.getElementById("model-title");
        this.description = document.getElementById("model-description");
        this.vertices = document.getElementById("model-vertices");
        this.faces = document.getElementById("model-faces");
        this.format = document.getElementById("model-format");
        this.tabs = Array.from(document.querySelectorAll("[data-model-id]"));
        this.wireframeButton = document.getElementById("viewer-wireframe");
        this.resetButton = document.getElementById("viewer-reset");
        this.loader = new GLTFLoader();
        this.model = null;
        this.wireframe = false;
        this.animationFrame = null;

        this.initScene();
        this.bindEvents();
        this.loadModel("flame-neutral");
    }

    initScene() {
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0xf6fbfa);

        const { width, height } = this.container.getBoundingClientRect();
        this.camera = new THREE.PerspectiveCamera(42, width / height, 0.01, 100);
        this.camera.position.set(0, 0.08, 2.4);

        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.setSize(width, height);
        this.renderer.outputColorSpace = THREE.SRGBColorSpace;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.08;
        this.container.appendChild(this.renderer.domElement);

        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.08;
        this.controls.minDistance = 0.8;
        this.controls.maxDistance = 5;

        const ambientLight = new THREE.AmbientLight(0xffffff, 0.78);
        this.scene.add(ambientLight);

        const hemiLight = new THREE.HemisphereLight(0xffffff, 0xd8e7fb, 1.35);
        this.scene.add(hemiLight);

        const keyLight = new THREE.DirectionalLight(0xffffff, 1.55);
        keyLight.position.set(2.4, 3, 4.5);
        this.scene.add(keyLight);

        const fillLight = new THREE.DirectionalLight(0xffffff, 0.85);
        fillLight.position.set(-2, 1.4, 3.2);
        this.scene.add(fillLight);

        const rimLight = new THREE.DirectionalLight(0xd8e7fb, 0.75);
        rimLight.position.set(-2.4, 1.2, -2);
        this.scene.add(rimLight);

        const grid = new THREE.GridHelper(2.4, 12, 0x88b8a7, 0xdfe8e7);
        grid.position.y = -0.72;
        this.scene.add(grid);

        window.addEventListener("resize", () => this.resize());
        this.animate();
    }

    bindEvents() {
        this.tabs.forEach((tab) => {
            tab.addEventListener("click", () => this.loadModel(tab.dataset.modelId));
        });

        this.resetButton?.addEventListener("click", () => this.frameModel());

        this.wireframeButton?.addEventListener("click", () => {
            this.wireframe = !this.wireframe;
            this.applyWireframe();
            this.wireframeButton.classList.toggle("is-active", this.wireframe);
        });
    }

    loadModel(id) {
        const modelInfo = MODELS[id];
        if (!modelInfo) {
            return;
        }

        this.setActiveTab(id);
        this.updateDetails(modelInfo);
        this.setStatus("Загрузка модели...");

        this.loader.load(
            modelInfo.url,
            (gltf) => {
                if (this.model) {
                    this.scene.remove(this.model);
                    this.disposeObject(this.model);
                }

                this.model = gltf.scene;
                this.model.traverse((child) => {
                    if (!child.isMesh) {
                        return;
                    }
                    child.castShadow = false;
                    child.receiveShadow = false;
                    if (child.geometry) {
                        child.geometry.computeVertexNormals();
                        child.geometry.normalizeNormals();
                    }
                    const preserveTexture =
                        (id === "flame-textured"
                            || id === "flame-detail-textured"
                            || id === "deca") && Boolean(child.material?.map);
                    const preserveVertexColors = id === "flame-mask-baseline";
                    if (!preserveTexture) {
                        child.material = new THREE.MeshPhongMaterial({
                            color: preserveVertexColors ? 0xffffff : 0xc98f78,
                            emissive: preserveVertexColors ? 0x181818 : 0x241510,
                            specular: 0x4a4a4a,
                            shininess: 24,
                            vertexColors: preserveVertexColors,
                            side: THREE.DoubleSide,
                        });
                    } else {
                        const textureMap = child.material.map;
                        if (textureMap) {
                            textureMap.colorSpace = THREE.SRGBColorSpace;
                            textureMap.needsUpdate = true;
                        }
                        child.material = new THREE.MeshPhongMaterial({
                            map: textureMap,
                            color: 0xffffff,
                            emissive: 0x1d1d1d,
                            specular: 0x505050,
                            shininess: 18,
                            side: THREE.DoubleSide,
                        });
                    }
                    child.material.needsUpdate = true;
                });

                this.scene.add(this.model);
                this.frameModel();
                this.applyWireframe();
                this.setStatus("");
            },
            undefined,
            (error) => {
                console.error(error);
                this.setStatus("Не удалось загрузить GLB");
            }
        );
    }

    frameModel() {
        if (!this.model) {
            return;
        }

        const box = new THREE.Box3().setFromObject(this.model);
        const size = box.getSize(new THREE.Vector3());
        const center = box.getCenter(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        const scale = maxDim > 0 ? 1.35 / maxDim : 1;

        this.model.scale.setScalar(scale);
        this.model.rotation.set(0, 0, 0);
        this.model.position.set(
            -center.x * scale,
            -center.y * scale + 0.12,
            -center.z * scale
        );

        this.controls.target.set(0, 0.08, 0);
        this.camera.position.set(0, 0.12, 2.15);
        this.camera.lookAt(0, 0.08, 0);
        this.controls.update();
    }

    applyWireframe() {
        if (!this.model) {
            return;
        }

        this.model.traverse((child) => {
            if (child.isMesh && child.material) {
                child.material.wireframe = this.wireframe;
                child.material.needsUpdate = true;
            }
        });
    }

    setActiveTab(id) {
        this.tabs.forEach((tab) => {
            tab.classList.toggle("is-active", tab.dataset.modelId === id);
        });
    }

    updateDetails(modelInfo) {
        this.title.textContent = modelInfo.title;
        this.description.textContent = modelInfo.description;
        this.vertices.textContent = modelInfo.vertices;
        this.faces.textContent = modelInfo.faces;
        this.format.textContent = modelInfo.format;
        this.download.href = modelInfo.url;
    }

    setStatus(message) {
        if (!this.status) {
            return;
        }

        this.status.textContent = message;
        this.status.hidden = !message;
    }

    resize() {
        const { width, height } = this.container.getBoundingClientRect();
        if (!width || !height) {
            return;
        }

        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }

    animate() {
        this.animationFrame = requestAnimationFrame(() => this.animate());
        if (this.model) {
            this.model.rotation.y += 0.0025;
        }
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }

    disposeObject(object) {
        object.traverse((child) => {
            if (child.geometry) {
                child.geometry.dispose();
            }
            if (child.material) {
                const materials = Array.isArray(child.material) ? child.material : [child.material];
                materials.forEach((material) => material.dispose());
            }
        });
    }
}

document.addEventListener("DOMContentLoaded", () => {
    const container = document.getElementById("face-viewer");
    if (container) {
        window.faceDemoViewer = new FaceDemoViewer(container);
    }
});
