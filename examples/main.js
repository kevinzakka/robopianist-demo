
import * as THREE           from 'three';
import { GUI              } from '../node_modules/three/examples/jsm/libs/lil-gui.module.min.js';
import { OrbitControls    } from '../node_modules/three/examples/jsm/controls/OrbitControls.js';
import { DragStateManager } from './utils/DragStateManager.js';
import  npyjs               from './utils/npy.js';
import { setupGUI, downloadExampleScenesFolder, loadSceneFromURL, getPosition, getQuaternion, toMujocoPos, standardNormal } from './mujocoUtils.js';
import   load_mujoco        from '../dist/mujoco_wasm.js';

const sampler = new Tone.Sampler({
  urls: {
    A1: "A1.mp3",
    A2: "A2.mp3",
    A3: "A3.mp3",
    A4: "A4.mp3",
    A5: "A5.mp3",
    A6: "A6.mp3",
    A7: "A7.mp3",
    C1: "C1.mp3",
    C2: "C2.mp3",
    C3: "C3.mp3",
    C4: "C4.mp3",
    C5: "C5.mp3",
    C6: "C6.mp3",
    C7: "C7.mp3",
    C8: "C8.mp3",
    "D#1": "Ds1.mp3",
    "D#2": "Ds2.mp3",
    "D#3": "Ds3.mp3",
    "D#4": "Ds4.mp3",
    "D#5": "Ds5.mp3",
    "D#6": "Ds6.mp3",
    "D#7": "Ds7.mp3",
    "F#1": "Fs1.mp3",
    "F#2": "Fs2.mp3",
    "F#3": "Fs3.mp3",
    "F#4": "Fs4.mp3",
    "F#5": "Fs5.mp3",
    "F#6": "Fs6.mp3",
    "F#7": "Fs7.mp3",
	},
	baseUrl: "https://tonejs.github.io/audio/salamander/",
}).toDestination();
let prevActivated = new Array(88).fill(false);

const key2note = new Map([
  [0, 'A0'],
  [1, 'A#0'],
  [2, 'B0'],
  [3, 'C1'],
  [4, 'C#1'],
  [5, 'D1'],
  [6, 'D#1'],
  [7, 'E1'],
  [8, 'F1'],
  [9, 'F#1'],
  [10, 'G1'],
  [11, 'G#1'],
  [12, 'A1'],
  [13, 'A#1'],
  [14, 'B1'],
  [15, 'C2'],
  [16, 'C#2'],
  [17, 'D2'],
  [18, 'D#2'],
  [19, 'E2'],
  [20, 'F2'],
  [21, 'F#2'],
  [22, 'G2'],
  [23, 'G#2'],
  [24, 'A2'],
  [25, 'A#2'],
  [26, 'B2'],
  [27, 'C3'],
  [28, 'C#3'],
  [29, 'D3'],
  [30, 'D#3'],
  [31, 'E3'],
  [32, 'F3'],
  [33, 'F#3'],
  [34, 'G3'],
  [35, 'G#3'],
  [36, 'A3'],
  [37, 'A#3'],
  [38, 'B3'],
  [39, 'C4'],
  [40, 'C#4'],
  [41, 'D4'],
  [42, 'D#4'],
  [43, 'E4'],
  [44, 'F4'],
  [45, 'F#4'],
  [46, 'G4'],
  [47, 'G#4'],
  [48, 'A4'],
  [49, 'A#4'],
  [50, 'B4'],
  [51, 'C5'],
  [52, 'C#5'],
  [53, 'D5'],
  [54, 'D#5'],
  [55, 'E5'],
  [56, 'F5'],
  [57, 'F#5'],
  [58, 'G5'],
  [59, 'G#5'],
  [60, 'A5'],
  [61, 'A#5'],
  [62, 'B5'],
  [63, 'C6'],
  [64, 'C#6'],
  [65, 'D6'],
  [66, 'D#6'],
  [67, 'E6'],
  [68, 'F6'],
  [69, 'F#6'],
  [70, 'G6'],
  [71, 'G#6'],
  [72, 'A6'],
  [73, 'A#6'],
  [74, 'B6'],
  [75, 'C7'],
  [76, 'C#7'],
  [77, 'D7'],
  [78, 'D#7'],
  [79, 'E7'],
  [80, 'F7'],
  [81, 'F#7'],
  [82, 'G7'],
  [83, 'G#7'],
  [84, 'A7'],
  [85, 'A#7'],
  [86, 'B7'],
  [87, 'C8']
]);

// Load the MuJoCo Module
const mujoco = await load_mujoco();

// Set up Emscripten's Virtual File System
var initialScene = "piano_with_shadow_hands/scene.xml";
mujoco.FS.mkdir('/working');
mujoco.FS.mount(mujoco.MEMFS, { root: '.' }, '/working');

export class RoboPianistDemo {
  constructor() {
    this.mujoco = mujoco;

    // Activate Audio upon first interaction
    document.addEventListener('pointerdown', () => { if (Tone.context.state !== "running") { Tone.context.resume(); } });

    // Define Random State Variables
    this.params = { scene: initialScene, paused: false, help: false, ctrlnoiserate: 0.0, ctrlnoisestd: 0.0, keyframeNumber: 0 };
    this.mujoco_time = 0.0;
    this.bodies  = {}, this.lights = {};
    this.tmpVec  = new THREE.Vector3();
    this.tmpQuat = new THREE.Quaternion();
    this.updateGUICallbacks = [];

    this.container = document.createElement( 'div' );
    document.body.appendChild( this.container );

    this.scene = new THREE.Scene();
    this.scene.name = 'scene';

    this.camera = new THREE.PerspectiveCamera( 45, window.innerWidth / window.innerHeight, 0.001, 100 );
    this.camera.name = 'PerspectiveCamera';
    this.camera.position.set( -0.6, 0.7, 0.0 );
    this.scene.add(this.camera);

    this.scene.background = new THREE.Color(0.15, 0.25, 0.35);
    this.scene.fog = new THREE.Fog(this.scene.background, 15, 25.5 );

    this.ambientLight = new THREE.AmbientLight( 0xffffff, 0.1 );
    this.ambientLight.name = 'AmbientLight';
    this.scene.add( this.ambientLight );

    this.renderer = new THREE.WebGLRenderer( { antialias: true } );
    this.renderer.setPixelRatio( window.devicePixelRatio );
    this.renderer.setSize( window.innerWidth, window.innerHeight );
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap; // default THREE.PCFShadowMap
    this.renderer.setAnimationLoop( this.render.bind(this) );

    this.container.appendChild( this.renderer.domElement );

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, 0.7, 0);
    this.controls.panSpeed = 2;
    this.controls.zoomSpeed = 1;
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.10;
    this.controls.screenSpacePanning = true;
    this.controls.update();

    window.addEventListener('resize', this.onWindowResize.bind(this));

    // Initialize the Drag State Manager.
    this.dragStateManager = new DragStateManager(this.scene, this.renderer, this.camera, this.container.parentElement, this.controls);
  }

  async init() {
    // Download the the examples to MuJoCo's virtual file system
    await downloadExampleScenesFolder(mujoco);

    // Initialize the three.js Scene using the .xml Model in initialScene
    [this.model, this.state, this.simulation, this.bodies, this.lights] =
      await loadSceneFromURL(mujoco, initialScene, this);

    this.gui = new GUI();
    setupGUI(this);

    this.npyjs = new npyjs();
    this.npyjs.load("./examples/scenes/piano_with_shadow_hands/twinkle_twinkle_actions.npy", (loaded) => {
      this.pianoControl = loaded;
      console.log(this.pianoControl);
      this.currentFrame = 0;
    });
  }

  onWindowResize() {
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize( window.innerWidth, window.innerHeight );
  }

  processPianoState() {
    let activation = new Array(88).fill(false);

    // Detect which keys are pressed and color them accordingly.
    for (let b = 0; b < this.model.nbody(); b++) {
      if (this.bodies[b]) {
        if (this.bodies[b].name.indexOf("piano/") < 0) { continue; }
        if (this.bodies[b].name.indexOf("key") < 0) { continue; }
        let jnt_adr = this.model.body_jntadr()[b];
        if (jnt_adr < 0) { continue; }
        let qpos_adr = this.model.jnt_qposadr()[jnt_adr];
        let qpos = this.simulation.qpos()[qpos_adr];
        let qpos_min = this.model.jnt_range()[2*jnt_adr + 0];
        let qpos_max = this.model.jnt_range()[2*jnt_adr + 1];
        let qpos_state = Math.max(qpos_min, Math.min(qpos, qpos_max));
        if (Math.abs(qpos_state - qpos_max) <= 0.00872665) {
          let key = parseInt(this.bodies[b].name.split("_")[2]);
          activation[key] = true;
          this.bodies[b].children[0].material.color.setRGB(0.2, 0.8, 0.2);
        } else {
          if (this.bodies[b].name.indexOf("white") >= 0) {
            this.bodies[b].children[0].material.color.setRGB(0.9, 0.9, 0.9);
          } else {
            this.bodies[b].children[0].material.color.setRGB(0.1, 0.1, 0.1);
          }
        }
      }
    }

    // xor the current activation with the previous activation.
    let state_change = new Array(88).fill(false);
    for (let i = 0; i < 88; i++) {
      state_change[i] = activation[i] ^ prevActivated[i];
    }

    // Note on events.
    for (let i = 0; i < 88; i++) {
      if (state_change[i] && !prevActivated[i]) {
        let note = key2note.get(i);
        console.log(note);
        sampler.triggerAttack(note);
      }
    }

    // Note off events.
    for (let i = 0; i < 88; i++) {
      if (state_change[i] && !activation[i]) {
        let note = key2note.get(i);
        sampler.triggerRelease(note);
      }
    }

    // Update the previous activation.
    for (let i = 0; i < 88; i++) {
      prevActivated[i] = activation[i];
    }
  }

  render(timeMS) {
    this.controls.update();

    // Return if the model hasn't been loaded yet
    if (!this.model) { return; }

    if (!this.params["paused"]) {
      let timestep = this.model.getOptions().timestep;
      if (timeMS - this.mujoco_time > 35.0) { this.mujoco_time = timeMS; }
      while (this.mujoco_time < timeMS) {

        // Jitter the control state with gaussian random noise
        if (this.pianoControl) {
          let currentCtrl = this.simulation.ctrl();
          for (let i = 0; i < currentCtrl.length; i++) {
            // Play one control frame every 100ms
            currentCtrl[i] = this.pianoControl.data[
              (currentCtrl.length * Math.floor(this.mujoco_time / 50)) + i];
            this.params["Actuator " + i] = currentCtrl[i];
          }
        }

        if (this.params["ctrlnoisestd"] > 0.0) {
          let rate  = Math.exp(-timestep / Math.max(1e-10, this.params["ctrlnoiserate"]));
          let scale = this.params["ctrlnoisestd"] * Math.sqrt(1 - rate * rate);
          let currentCtrl = this.simulation.ctrl();
          for (let i = 0; i < currentCtrl.length; i++) {
            currentCtrl[i] = rate * currentCtrl[i] + scale * standardNormal();
            this.params["Actuator " + i] = currentCtrl[i];
          }
        }

        // Clear old perturbations, apply new ones.
        for (let i = 0; i < this.simulation.qfrc_applied().length; i++) { this.simulation.qfrc_applied()[i] = 0.0; }
        let dragged = this.dragStateManager.physicsObject;
        if (dragged && dragged.bodyID) {
          for (let b = 0; b < this.model.nbody(); b++) {
            if (this.bodies[b]) {
              getPosition  (this.simulation.xpos (), b, this.bodies[b].position);
              getQuaternion(this.simulation.xquat(), b, this.bodies[b].quaternion);
              this.bodies[b].updateWorldMatrix();
            }
          }
          let bodyID = dragged.bodyID;
          this.dragStateManager.update(); // Update the world-space force origin
          let force = toMujocoPos(this.dragStateManager.currentWorld.clone().sub(this.dragStateManager.worldHit).multiplyScalar(this.model.body_mass()[bodyID] * 250));
          let point = toMujocoPos(this.dragStateManager.worldHit.clone());
          this.simulation.applyForce(force.x, force.y, force.z, 0, 0, 0, point.x, point.y, point.z, bodyID);

          // TODO: Apply pose perturbations (mocap bodies only).
        }

        this.simulation.step();
        if (this.params.scene == "piano_with_shadow_hands/scene.xml") { this.processPianoState(); }

        this.mujoco_time += timestep * 1000.0;
      }

    } else if (this.params["paused"]) {
      this.dragStateManager.update(); // Update the world-space force origin
      let dragged = this.dragStateManager.physicsObject;
      if (dragged && dragged.bodyID) {
        let b = dragged.bodyID;
        getPosition  (this.simulation.xpos (), b, this.tmpVec , false); // Get raw coordinate from MuJoCo
        getQuaternion(this.simulation.xquat(), b, this.tmpQuat, false); // Get raw coordinate from MuJoCo

        let offset = toMujocoPos(this.dragStateManager.currentWorld.clone()
          .sub(this.dragStateManager.worldHit).multiplyScalar(0.3));
        if (this.model.body_mocapid()[b] >= 0) {
          // Set the root body's mocap position...
          console.log("Trying to move mocap body", b);
          let addr = this.model.body_mocapid()[b] * 3;
          let pos  = this.simulation.mocap_pos();
          pos[addr+0] += offset.x;
          pos[addr+1] += offset.y;
          pos[addr+2] += offset.z;
        } else {
          // Set the root body's position directly...
          let root = this.model.body_rootid()[b];
          let addr = this.model.jnt_qposadr()[this.model.body_jntadr()[root]];
          let pos  = this.simulation.qpos();
          pos[addr+0] += offset.x;
          pos[addr+1] += offset.y;
          pos[addr+2] += offset.z;
        }
      }

      this.simulation.forward();
      sampler.releaseAll();
    }

    // Update body transforms.
    for (let b = 0; b < this.model.nbody(); b++) {
      if (this.bodies[b]) {
        getPosition  (this.simulation.xpos (), b, this.bodies[b].position);
        getQuaternion(this.simulation.xquat(), b, this.bodies[b].quaternion);
        this.bodies[b].updateWorldMatrix();
      }
    }

    // Update light transforms.
    for (let l = 0; l < this.model.nlight(); l++) {
      if (this.lights[l]) {
        getPosition(this.simulation.light_xpos(), l, this.lights[l].position);
        getPosition(this.simulation.light_xdir(), l, this.tmpVec);
        this.lights[l].lookAt(this.tmpVec.add(this.lights[l].position));
      }
    }

    // Render!
    this.renderer.render( this.scene, this.camera );
  }
}

let demo = new RoboPianistDemo();
await demo.init();
