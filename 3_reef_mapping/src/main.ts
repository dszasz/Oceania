import "./style.css";
import * as THREE from "three";
import Papa from "papaparse";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";


// ============================================================
// 1. DATA TYPES
// ============================================================

type Track = {
    Track_ID: string;
    Title: string;
    Artist: string;
    Genre: string;
    Subgenre1: string;
    Subgenre2: string;
    Mood1: string;
    Mood2: string;
    Light: string;
    BPM: number;
    Camelot: string;
    YouTube_URL?: string;
    x: number;
    y: number;
    z: number;
};

type Connection = {
    Track_A: string;
    Track_B: string;
    BPM_similarity: number;
    Key_similarity: number;
    Connection_score: number;
};

type Metadata = {
    Track_ID: string;
    YouTube_URL: string;
};


// ============================================================
// 2. GLOBAL VARIABLES FOR CLICKING
// ============================================================

const clickableTracks: THREE.Mesh[] = [];

let selectedSphere: THREE.Mesh | null = null;


// ============================================================
// 3. LOAD CSV FILE
// ============================================================

async function loadCSV<T>(path: string): Promise<T[]> {

    const response = await fetch(path);

    const text = await response.text();

    const result = Papa.parse<T>(text, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true
    });

    return result.data;
}


// ============================================================
// 4. THREE.JS SCENE
// ============================================================

const scene = new THREE.Scene();

scene.background = new THREE.Color(0x050510);


// ============================================================
// 5. CAMERA
// ============================================================

const camera = new THREE.PerspectiveCamera(
    60,
    window.innerWidth / window.innerHeight,
    0.1,
    1000
);

camera.position.set(
    50,
    90,
    140
);


// ============================================================
// 6. RENDERER
// ============================================================

const renderer = new THREE.WebGLRenderer({
    antialias: true
});

renderer.setSize(
    window.innerWidth,
    window.innerHeight
);

document.body.appendChild(
    renderer.domElement
);


// ============================================================
// 7. TRACK INFORMATION PANEL
// ============================================================

const infoPanel = document.createElement("div");

infoPanel.style.position = "absolute";
infoPanel.style.top = "20px";
infoPanel.style.left = "20px";
infoPanel.style.padding = "15px 20px";
infoPanel.style.background = "rgba(0, 0, 0, 0.78)";
infoPanel.style.color = "white";
infoPanel.style.fontFamily = "Arial, sans-serif";
infoPanel.style.borderRadius = "8px";
infoPanel.style.display = "none";
infoPanel.style.pointerEvents = "auto";
infoPanel.style.minWidth = "220px";
infoPanel.style.lineHeight = "1.5";

document.body.appendChild(infoPanel);


// Prevent clicks inside the info panel from being treated
// as clicks on empty 3D space

infoPanel.addEventListener(
    "click",
    (event) => {
        event.stopPropagation();
    }
);


// ============================================================
// 8. CAMERA CONTROLS
// ============================================================

const controls = new OrbitControls(
    camera,
    renderer.domElement
);

controls.target.set(
    50,
    20,
    50
);

controls.enableDamping = true;


// ============================================================
// 9. LIGHT
// ============================================================

const ambientLight = new THREE.AmbientLight(
    0xffffff,
    1.5
);

scene.add(ambientLight);


// ============================================================
// 10. GRID
// ============================================================

const grid = new THREE.GridHelper(
    100,
    10
);

grid.position.set(
    50,
    0,
    50
);

scene.add(grid);


// ============================================================
// 11. GENRE COLORS
// ============================================================

const genreColors: Record<string, number> = {

    Techno: 0x00ffff,

    House: 0xff00ff,

    Electronic: 0xffff00

};

function getGenreColor(genre: string): number {

    return genreColors[genre] ?? 0xffffff;

}


// ============================================================
// 12. CREATE TRACK SPHERES
// ============================================================

function createTracks(
    tracks: Track[]
): Map<string, THREE.Mesh> {

    const trackObjects =
        new Map<string, THREE.Mesh>();


    for (const track of tracks) {

        const geometry =
            new THREE.SphereGeometry(
                1.2,
                24,
                24
            );


        const material =
            new THREE.MeshStandardMaterial({
                color: getGenreColor(track.Genre)
            });


        const sphere =
            new THREE.Mesh(
                geometry,
                material
            );


        /*
        Python coordinates:

        x = reef x
        y = reef y
        z = normalized BPM

        Three.js uses Y as vertical.

        Therefore:

        Python x -> Three.js x
        Python z -> Three.js y
        Python y -> Three.js z
        */

        sphere.position.set(

            track.x,

            track.z * 0.35,

            track.y

        );


        // Store all metadata directly on sphere
        sphere.userData = track;


        scene.add(sphere);


        // Make sphere clickable
        clickableTracks.push(sphere);


        // Store sphere by Track_ID
        trackObjects.set(
            track.Track_ID,
            sphere
        );

    }


    return trackObjects;
}


// ============================================================
// 13. CREATE CONNECTION LINES
// ============================================================

function createConnections(

    connections: Connection[],

    trackObjects:
        Map<string, THREE.Mesh>

) {

    for (const connection of connections) {

        const objectA =
            trackObjects.get(
                connection.Track_A
            );

        const objectB =
            trackObjects.get(
                connection.Track_B
            );


        if (!objectA || !objectB) {
            continue;
        }

        const geometry =
            new THREE.BufferGeometry()
                .setFromPoints([
                    objectA.position,
                    objectB.position
                ]);


        const material =
            new THREE.LineBasicMaterial({

                color: 0xffffff,

                transparent: true,

                opacity:
                    0.08 +
                    connection.Connection_score
                    * 0.25

            });


        const line =
            new THREE.Line(
                geometry,
                material
            );


        scene.add(line);

    }

}


// ============================================================
// 14. CLICK DETECTION
// ============================================================

const raycaster = new THREE.Raycaster();

const mouse = new THREE.Vector2();


window.addEventListener(
    "click",
    (event) => {


        // Convert mouse position to Three.js coordinates

        mouse.x =
            (event.clientX / window.innerWidth)
            * 2 - 1;

        mouse.y =
            -(event.clientY / window.innerHeight)
            * 2 + 1;


        raycaster.setFromCamera(
            mouse,
            camera
        );


        // Check whether a sphere was clicked

        const intersections =
            raycaster.intersectObjects(
                clickableTracks
            );


        // ====================================================
        // CLICKED A TRACK
        // ====================================================

        if (intersections.length > 0) {

            const sphere =
                intersections[0]
                    .object as THREE.Mesh;


            const track =
                sphere.userData as Track;


            // Reset previously selected sphere

            if (selectedSphere) {

                selectedSphere.scale.set(
                    1,
                    1,
                    1
                );

            }


            // Select clicked sphere

            selectedSphere = sphere;


            // Make selected sphere larger

            selectedSphere.scale.set(
                1.8,
                1.8,
                1.8
            );


            // Show track information

            infoPanel.innerHTML = `

                <strong style="
                    font-size: 20px;
                ">
                    ${track.Title}
                </strong>

                <br>

                <span style="
                    font-size: 16px;
                ">
                    ${track.Artist}
                </span>

                <br><br>

                <strong>Genre:</strong>
                ${track.Genre}

                <br>

                <strong>Subgenre:</strong>
                ${track.Subgenre1}

                <br>

                <strong>BPM:</strong>
                ${track.BPM}

                <br>

                <strong>Key:</strong>
                ${track.Camelot}

                <br><br>

                ${
                    track.YouTube_URL
                    ? `
                        <a
                            href="${track.YouTube_URL}"
                            target="_blank"
                            rel="noopener noreferrer"
                            style="
                                display:inline-block;
                                padding:8px 14px;
                                background:#ffffff;
                                color:#000000;
                                text-decoration:none;
                                border-radius:6px;
                                font-weight:bold;
                            "
                        >
                            ▶ Play on YouTube
                        </a>
                    `
                    : `
                        <span style="
                            color:#999;
                            font-size:13px;
                        ">
                            No YouTube link available
                        </span>
                    `
                }

            `;


            infoPanel.style.display =
                "block";

        }


        // ====================================================
        // CLICKED EMPTY SPACE
        // ====================================================

        else {

            if (selectedSphere) {

                selectedSphere.scale.set(
                    1,
                    1,
                    1
                );

                selectedSphere = null;

            }


            infoPanel.style.display =
                "none";

        }

    }
);


// ============================================================
// 15. LOAD AND MERGE DATA
// ============================================================

async function initialise() {

    // Main coordinates file

    const tracks =
        await loadCSV<Track>(
            "/data/track_coordinates.csv"
        );


    // Connection file

    const connections =
        await loadCSV<Connection>(
            "/data/connections.csv"
        );


    // Original metadata file containing YouTube URLs

    const metadata =
        await loadCSV<Metadata>(
            "/data/PL_01_final.csv"
        );


    // ========================================================
    // CREATE YOUTUBE LOOKUP TABLE
    // ========================================================

    /*
    Creates:

    Track_ID -> YouTube URL

    Example:

    PL01_0001 -> https://youtube.com/...
    PL01_0002 -> https://youtube.com/...
    */

    const youtubeLookup =
        new Map<string, string>();


    for (const row of metadata) {

        if (
            row.Track_ID &&
            row.YouTube_URL
        ) {

            youtubeLookup.set(
                String(row.Track_ID).trim(),
                String(row.YouTube_URL).trim()
            );

        }

    }


    // ========================================================
    // MERGE YOUTUBE URLs INTO TRACK DATA
    // ========================================================

    for (const track of tracks) {

        const trackID =
            String(track.Track_ID).trim();


        track.YouTube_URL =
            youtubeLookup.get(
                trackID
            );

    }


    // ========================================================
    // DEBUGGING
    // ========================================================

    console.log(
        "Metadata:",
        metadata
    );


    console.log(
        "YouTube lookup:",
        youtubeLookup
    );


    console.log(
        "Tracks with YouTube URLs:",
        tracks
    );


    console.log(
        "Connections:",
        connections
    );


    // ========================================================
    // CREATE VISUALIZATION
    // ========================================================

    const trackObjects =
        createTracks(
            tracks
        );


    createConnections(
        connections,
        trackObjects
    );

}


initialise();


// ============================================================
// 16. ANIMATION LOOP
// ============================================================

function animate() {

    requestAnimationFrame(
        animate
    );


    controls.update();


    renderer.render(
        scene,
        camera
    );

}


animate();


// ============================================================
// 17. WINDOW RESIZING
// ============================================================

window.addEventListener(
    "resize",
    () => {

        camera.aspect =
            window.innerWidth
            /
            window.innerHeight;


        camera.updateProjectionMatrix();


        renderer.setSize(
            window.innerWidth,
            window.innerHeight
        );

    }
);