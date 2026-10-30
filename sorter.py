#!/usr/bin/env python3
"""Camera-color AI parcel sorter. Deterministic CPU/PyBullet DIRECT demo."""
from __future__ import annotations
import argparse, csv, json, math, os, random, time
from pathlib import Path

import cv2
import imageio.v2 as imageio
import joblib
import numpy as np
import pybullet as p
import pybullet_data
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 20261029
LABELS = ["RED", "GREEN", "BLUE"]
STARTS = [(0.32, -0.25), (0.43, -0.25), (0.54, -0.25)]
COLORS = {
    "RED": (0.88, 0.12, 0.10, 1),
    "GREEN": (0.10, 0.72, 0.20, 1),
    "BLUE": (0.10, 0.30, 0.92, 1),
}
# Slightly muted matching tray colors; not used as classifier input.
TRAY_COLORS = {"RED": (0.55, 0.20, 0.18, 1), "GREEN": (0.18, 0.48, 0.24, 1), "BLUE": (0.18, 0.28, 0.58, 1)}
W, H, FPS = 1280, 720, 15
RENDER_W, RENDER_H = 640, 360  # Native simulator raster, upscaled only for readable video overlays.
DT = 1.0 / 240.0
RENDER_EVERY = 16  # 16 physics steps per video frame => 15 fps at nominal 1x simulation time.
MOTION_TIME_SCALE = 1.8  # Slower simulated motor trajectories, not video slow motion.
CAM_TARGET = (0.52, 0.0, 0.57)
CAM_EYE = (1.35, -1.45, 2.30)
FOV = 54

class Sorter:
    def __init__(self, out: Path, record: bool):
        self.out, self.record = out, record
        self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(DT, physicsClientId=self.client)
        p.setPhysicsEngineParameter(fixedTimeStep=DT, numSolverIterations=100, enableFileCaching=0, physicsClientId=self.client)
        self.view = p.computeViewMatrix(CAM_EYE, CAM_TARGET, [0, 0, 1])
        self.vision_view = p.computeViewMatrix((0.50, -0.20, 2.80), (0.50, -0.20, 0.60), [0, 1, 0])
        self.proj = p.computeProjectionMatrixFOV(FOV, W/H, 0.1, 5.0)
        self.writer = None
        self.frame_count = 0
        self.sim_steps = 0
        self.frame_cache = None
        self.live_crop = None
        self.objects = {}
        self.tray_centers = {"RED": (0.50, -0.50), "GREEN": (0.70, 0.0), "BLUE": (0.50, 0.50)}
        self._scene()

    def _box(self, pos, half, rgba, mass=0, friction=0.8):
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=self.client)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=rgba, physicsClientId=self.client)
        bid = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=pos, physicsClientId=self.client)
        p.changeDynamics(bid, -1, lateralFriction=friction, restitution=0.0, physicsClientId=self.client)
        return bid

    def _scene(self):
        plane = p.loadURDF("plane.urdf", [0, 0, -0.015], useFixedBase=True, physicsClientId=self.client)
        p.changeVisualShape(plane, -1, rgbaColor=(0.20,0.24,0.29,1), textureUniqueId=-1, physicsClientId=self.client)
        # Custom workbench: tabletop top is z=0.60 m.
        self._box((0.48, 0, 0.56), (0.72, 0.70, 0.04), (0.72, 0.75, 0.78, 1))
        # Four simple legs make the table read clearly in the camera image.
        for x in (-0.13, 1.09):
            for y in (-0.56, 0.56):
                self._box((x, y, 0.26), (0.035, 0.035, 0.26), (0.26, 0.30, 0.34, 1))
        # Shallow, physically collidable receiving trays: floor and three walls.
        self.trays = {}
        for label, (x, y) in self.tray_centers.items():
            col = TRAY_COLORS[label]
            parts = [
                ((x, y, 0.615), (0.105, 0.105, 0.015)),
                ((x-0.105, y, 0.675), (0.012, 0.105, 0.060)),
                ((x+0.105, y, 0.675), (0.012, 0.105, 0.060)),
                ((x, y-0.105, 0.675), (0.105, 0.012, 0.060)),
                ((x, y+0.105, 0.675), (0.105, 0.012, 0.060)),
            ]
            self.trays[label] = [self._box(pos, half, col) for pos, half in parts]
        # Franka Panda model from pybullet_data; own scene and controller.
        # Place the fixed robot base on top of the 0.60 m workbench; placing it at
        # world z=0 would intersect the tabletop and produce nonphysical launches.
        self.robot = p.loadURDF("franka_panda/panda.urdf", basePosition=(0, 0, 0.60), useFixedBase=True, flags=p.URDF_USE_INERTIA_FROM_FILE, physicsClientId=self.client)
        self.arm_joints = list(range(7))
        self.ee_link = 11  # panda_grasptarget: aim at the center between fingers, not the hand origin.
        self.fingers = [9, 10]
        self.lower, self.upper, self.ranges = [], [], []
        self.rest = [0.0, -0.4, 0.0, -1.8, 0.0, 1.5, 0.8]
        for j in self.arm_joints:
            inf = p.getJointInfo(self.robot, j, physicsClientId=self.client)
            self.lower.append(inf[8]); self.upper.append(inf[9]); self.ranges.append(inf[9]-inf[8])
        for j, q in zip(self.arm_joints, self.rest):
            p.resetJointState(self.robot, j, q, physicsClientId=self.client)
            p.setJointMotorControl2(self.robot, j, p.POSITION_CONTROL, targetPosition=q, force=100, physicsClientId=self.client)
        for j in self.fingers:
            p.resetJointState(self.robot, j, 0.04, physicsClientId=self.client)
            p.setJointMotorControl2(self.robot, j, p.POSITION_CONTROL, targetPosition=0.04, force=35, physicsClientId=self.client)
        self.obj_ids = {}
        # Initial queue, clear of tray footprints and reachable work envelope.
        for label, (x, y) in zip(LABELS, STARTS):
            oid = self.make_parcel(label, (x, y, 0.626))
            self.objects[label] = oid

    def make_parcel(self, label, xyz, variant=0):
        color = COLORS[label]
        # training/test color nuisance variation is restricted so the class remains unambiguous.
        if variant:
            rng = np.random.default_rng(variant)
            rgb = np.clip(np.array(color[:3]) * rng.uniform(0.82, 1.14, 3) + rng.uniform(-0.025, 0.025, 3), 0.03, 0.98)
            color = (*rgb.tolist(), 1)
        half = (0.027, 0.027, 0.027)
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=self.client)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=color, physicsClientId=self.client)
        bid = p.createMultiBody(baseMass=0.16, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=xyz, physicsClientId=self.client)
        p.changeDynamics(bid, -1, lateralFriction=0.9, rollingFriction=0.01, restitution=0.0, physicsClientId=self.client)
        return bid

    def camera(self, vision=False):
        img = p.getCameraImage(RENDER_W, RENDER_H, self.vision_view if vision else self.view, self.proj, renderer=p.ER_TINY_RENDERER,
                               flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX, physicsClientId=self.client)
        rgba = np.asarray(img[2], dtype=np.uint8).reshape(RENDER_H, RENDER_W, 4)
        seg = np.asarray(img[4], dtype=np.int32).reshape(RENDER_H, RENDER_W)
        rgb = rgba[:, :, :3].copy()
        if not vision:
            rgb = cv2.resize(rgb,(W,H),interpolation=cv2.INTER_LINEAR)
            seg = cv2.resize(seg,(W,H),interpolation=cv2.INTER_NEAREST)
        self.frame_cache = (rgb, seg)
        return self.frame_cache

    @staticmethod
    def crop_for_body(rgb, seg, body_id):
        # Simulator segmentation is privileged ground truth used only to localize/crop;
        # it is never fed to the classifier. The model sees RGB camera pixels only.
        mask = (seg & ((1 << 24) - 1)) == int(body_id)
        yy, xx = np.where(mask)
        if len(xx) < 10:
            return None
        pad = 5
        rh,rw=rgb.shape[:2]
        x0, x1 = max(0, xx.min()-pad), min(rw, xx.max()+pad+1)
        y0, y1 = max(0, yy.min()-pad), min(rh, yy.max()+pad+1)
        crop = rgb[y0:y1, x0:x1].copy()
        crop[~mask[y0:y1, x0:x1]] = 0
        return crop

    @staticmethod
    def features(crop):
        # Privileged simulator mask isolates RGB pixels, not the class label.
        # Robust channel statistics reduce background/pose overfitting.
        pixels = crop[np.any(crop != 0, axis=2)].astype(np.float32) / 255.0
        if not len(pixels): raise RuntimeError('empty foreground RGB crop')
        return np.concatenate([pixels.mean(axis=0), pixels.std(axis=0),
                               np.percentile(pixels, [25,50,75], axis=0).reshape(-1)]).astype(np.float32)

    def collect_dataset(self, n_per_class=60):
        X, y, split = [], [], []
        samples = []
        crop_dir = self.out/'dataset'/'crops'
        crop_dir.mkdir(parents=True,exist_ok=True)
        # Explicit train/test render split by independently randomized pose/color nuisance.
        # Test crops are separately rendered and never used in fit().
        original = {lab: self.objects[lab] for lab in LABELS}
        for lab in LABELS:
            p.removeBody(self.objects[lab], physicsClientId=self.client)
        rng = np.random.default_rng(SEED)
        for label_id, label in enumerate(LABELS):
            for k in range(n_per_class):
                is_test = k >= int(n_per_class * 0.75)
                j = label_id*n_per_class+k
                # Keep learning objects on a visible, tray-free foreground strip;
                # both splits still use randomized, held-out coordinates and renders.
                x = float(rng.uniform(0.28, 0.56))
                y0 = float(rng.uniform(-0.30, -0.24))
                # random placement and tiny orientation variations, from actual physics objects
                oid = self.make_parcel(label, (x, y0, 0.630), variant=j+1)
                p.resetBasePositionAndOrientation(oid, (x,y0,0.630), p.getQuaternionFromEuler((0,0,float(rng.uniform(-math.pi,math.pi)))), physicsClientId=self.client)
                for _ in range(12): p.stepSimulation(physicsClientId=self.client)
                rgb, seg = self.camera(vision=True)
                crop = self.crop_for_body(rgb, seg, oid)
                if crop is None:
                    raise RuntimeError(f"カメラcrop取得失敗: {label} sample {k}")
                filename=f'{label.lower()}_{k:03}.png'
                cv2.imwrite(str(crop_dir/filename),cv2.cvtColor(crop,cv2.COLOR_RGB2BGR))
                samples.append({'file':f'crops/{filename}','class':label,'split':'test' if is_test else 'train','initial_xy_m':[x,y0],'color_variant_seed':j+1})
                X.append(self.features(crop)); y.append(label_id); split.append("test" if is_test else "train")
                p.removeBody(oid, physicsClientId=self.client)
        # Restore demo parcels at their initial positions.
        self.objects = {}
        for lab, pos in zip(LABELS, STARTS): self.objects[lab] = self.make_parcel(lab, (*pos, 0.626))
        for _ in range(120): p.stepSimulation(physicsClientId=self.client)
        (self.out/'dataset'/'manifest.json').write_text(json.dumps(samples,indent=2),encoding='utf-8')
        np.savez_compressed(self.out/'dataset'/'features.npz',X=np.asarray(X,np.float32),y=np.asarray(y),split=np.asarray(split))
        return np.asarray(X, np.float32), np.asarray(y), np.asarray(split)

    def fit_and_evaluate(self, model_path):
        X, y, split = self.collect_dataset()
        tr, te = split == "train", split == "test"
        model = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(16, 8), activation="relu", solver="lbfgs", alpha=0.01, max_iter=500, random_state=SEED, tol=1e-5))
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        labels = [int(v) for v in range(3)]
        cm = confusion_matrix(y[te], pred, labels=labels)
        report = classification_report(y[te], pred, labels=labels, target_names=LABELS, output_dict=True, zero_division=0)
        result = {
            "seed": SEED, "model": "sklearn MLPClassifier(hidden_layer_sizes=(16,8), solver=lbfgs)",
            "input": "15 RGB statistics (mean/std/25th/50th/75th percentile); simulator segmentation masks background, no body ID feature",
            "n_samples_total": int(len(y)), "train_n": int(tr.sum()), "test_n": int(te.sum()),
            "class_order": LABELS, "train_class_counts": {l:int(np.sum(y[tr]==i)) for i,l in enumerate(LABELS)},
            "test_class_counts": {l:int(np.sum(y[te]==i)) for i,l in enumerate(LABELS)},
            "test_correct": int(np.sum(pred == y[te])), "test_total": int(te.sum()),
            "test_accuracy": float(accuracy_score(y[te], pred)), "confusion_matrix_rows_true_cols_pred": cm.tolist(),
            "per_class": {l:{"precision":float(report[l]["precision"]),"recall":float(report[l]["recall"]),"f1":float(report[l]["f1-score"])} for l in LABELS},
            "iterations": int(model[-1].n_iter_), "stopped_before_max_iter": bool(model[-1].n_iter_ < 500),
            "limitations": ["Fixed camera and simulation appearance", "Privileged segmentation provides object crop; no camera-based 3D localization", "Synthetic rendered crops are not real-world data"]
        }
        joblib.dump(model, model_path)
        return model, result

    def ik(self, xyz):
        quat = p.getQuaternionFromEuler((math.pi, 0, 0))
        sol = p.calculateInverseKinematics(self.robot, self.ee_link, xyz, quat,
                    maxNumIterations=300, residualThreshold=1e-5, physicsClientId=self.client)
        return list(sol[:7])

    def goto(self, xyz, seconds=1.25, overlay=None):
        seconds *= MOTION_TIME_SCALE
        target = self.ik(xyz)
        cur = [p.getJointState(self.robot,j,physicsClientId=self.client)[0] for j in self.arm_joints]
        n = max(1, round(seconds / DT))
        for k in range(n):
            a = (k+1)/n
            # smoothstep joint-space path between IK waypoints
            s = a*a*(3-2*a)
            q = [(1-s)*u+s*v for u,v in zip(cur,target)]
            p.setJointMotorControlArray(self.robot, self.arm_joints, p.POSITION_CONTROL, targetPositions=q,
                                        forces=[100]*7, positionGains=[0.12]*7, velocityGains=[1.0]*7, physicsClientId=self.client)
            self._advance(overlay, record=True)

    def _advance(self, overlay=None, record=True):
        p.stepSimulation(physicsClientId=self.client); self.sim_steps += 1
        if self.record and record and self.sim_steps % RENDER_EVERY == 0:
            self._capture(overlay)

    def _capture(self, overlay=None):
        rgb, _ = self.camera()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        # English ASCII overlay for reliable glyph rendering across platforms.
        cv2.rectangle(bgr, (18, 16), (1262, 124), (14, 23, 34), -1)
        cv2.putText(bgr, "PHYSICAL AI  /  CAMERA-BASED PARCEL SORTER", (38, 52), cv2.FONT_HERSHEY_DUPLEX, 0.86, (245,248,250), 2, cv2.LINE_AA)
        cv2.putText(bgr, "PyBullet DIRECT | Panda IK | synthetic RGB MLP", (40, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (160,210,225), 1, cv2.LINE_AA)
        cv2.putText(bgr, "VIRTUAL GRASP: finger-contact checked + fixed constraint assist | crop localization uses simulator segmentation", (40, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (180,190,205), 1, cv2.LINE_AA)
        cv2.putText(bgr, f'SIM TIME {self.sim_steps*DT:05.1f}s | PLAYBACK 1x', (35, 152), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (35,40,45), 1, cv2.LINE_AA)
        if self.live_crop is not None:
            cv2.rectangle(bgr, (1038, 148), (1260, 370), (14,23,34), -1)
            cv2.putText(bgr, 'MLP INPUT / ORACLE MASK', (1047,172), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220,235,240), 1, cv2.LINE_AA)
            tile = cv2.resize(cv2.cvtColor(self.live_crop,cv2.COLOR_RGB2BGR),(164,164),interpolation=cv2.INTER_NEAREST)
            bgr[186:350,1067:1231] = tile
        if overlay:
            title = overlay.get("title", "")
            cv2.rectangle(bgr, (20, 620), (1260, 704), (14,23,34), -1)
            cv2.putText(bgr, title[:96], (40, 657), cv2.FONT_HERSHEY_DUPLEX, 0.68, (245,248,250), 1, cv2.LINE_AA)
            detail = overlay.get("detail", "")
            cv2.putText(bgr, detail[:145], (40, 687), cv2.FONT_HERSHEY_SIMPLEX, 0.49, (173,215,229), 1, cv2.LINE_AA)
        self.writer.append_data(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        self.frame_count += 1

    def move_and_record(self, xyz, seconds, title, detail):
        self.goto(xyz, seconds, {"title":title,"detail":detail})

    def set_gripper(self, opening, seconds, overlay):
        for j in self.fingers:
            p.setJointMotorControl2(self.robot, j, p.POSITION_CONTROL, targetPosition=opening, force=35, maxVelocity=0.12, physicsClientId=self.client)
        for _ in range(round(seconds/DT)):
            self._advance(overlay, record=True)

    def contact_verified(self, oid):
        cps = p.getContactPoints(bodyA=self.robot, bodyB=oid, physicsClientId=self.client)
        finger_contacts = [cp for cp in cps if cp[3] in self.fingers or cp[4] in self.fingers]
        objpos = p.getBasePositionAndOrientation(oid, physicsClientId=self.client)[0]
        ee = p.getLinkState(self.robot, self.ee_link, computeForwardKinematics=True, physicsClientId=self.client)[4]
        dist = float(np.linalg.norm(np.asarray(objpos)-np.asarray(ee)))
        return bool(finger_contacts) and dist <= 0.12, len(finger_contacts), dist

    def attach_virtual_grasp(self, oid):
        ok, ncontact, dist = self.contact_verified(oid)
        if not ok: return None, {"contact_verified":False,"finger_contact_count":ncontact,"ee_object_distance_m":dist,"constraint_created":False}
        parent_pos, parent_orn = p.getLinkState(self.robot, self.ee_link, computeForwardKinematics=True, physicsClientId=self.client)[4:6]
        obj_pos, obj_orn = p.getBasePositionAndOrientation(oid, physicsClientId=self.client)
        inv_pos, inv_orn = p.invertTransform(parent_pos, parent_orn)
        rel_pos, rel_orn = p.multiplyTransforms(inv_pos, inv_orn, obj_pos, obj_orn)
        cid = p.createConstraint(self.robot, self.ee_link, oid, -1, p.JOINT_FIXED, (0,0,0), rel_pos, (0,0,0), parentFrameOrientation=rel_orn, physicsClientId=self.client)
        p.changeConstraint(cid, maxForce=100, erp=0.9, physicsClientId=self.client)
        return cid, {"contact_verified":True,"finger_contact_count":ncontact,"ee_object_distance_m":dist,"constraint_created":True,"constraint_type":"fixed virtual-grasp assist; only after finger contact and <=0.12m EE/object distance"}

    def classify_object(self, model, label):
        rgb, seg = self.camera(vision=True)
        crop = self.crop_for_body(rgb, seg, self.objects[label])
        if crop is None: raise RuntimeError("live camera crop unavailable")
        probs = model.predict_proba(self.features(crop)[None,:])[0]
        idx = int(np.argmax(probs))
        return LABELS[idx], float(probs[idx]), crop

    def initial_frame(self, title, detail):
        if self.record: self._capture({"title":title,"detail":detail})

    def perform_sort(self, model):
        rows=[]
        # Classify all three objects from the live camera before the arm moves;
        # later motion can occlude objects, so reuse only those recorded inferences.
        inferences = {label: self.classify_object(model, label) for label in LABELS}
        # Small introductory real-physics settling period (displayed and logged).
        intro={"title":"LIVE SIMULATION | three parcels, three physically collidable trays","detail":"RGB MLP predicts category; predicted label causally selects the destination tray."}
        self.initial_frame(intro["title"], intro["detail"])
        for _ in range(round(2.0/DT)):
            self._advance(intro, record=True)
        for seq, actual in enumerate(LABELS,1):
            oid=self.objects[actual]
            # Oracle position is used only to aim and choose an instance crop via segmentation; never for class identity.
            predicted, confidence, crop=inferences[actual]
            self.live_crop = crop
            xyz, _=p.getBasePositionAndOrientation(oid,physicsClientId=self.client)
            z_table=0.60
            pre=(xyz[0],xyz[1],0.98)
            pick=(xyz[0],xyz[1],xyz[2]+0.003)
            dest=self.tray_centers[predicted]  # AI inference directly chooses target tray.
            tray_pre=(dest[0],dest[1],1.01)
            tray_place=(dest[0],dest[1],0.715)
            base={"seq":seq,"actual_class":actual,"predicted_class":predicted,"confidence":confidence,"target_tray":predicted,"object_body_id":int(oid),"oracle_pick_xyz_m":[float(v) for v in xyz],"oracle_pose_used_for_aiming":True,"camera_inferred_pose":False}
            note=f"LIVE RGB INFERENCE: {predicted}  confidence={confidence:.3f}  -> {predicted} TRAY"
            over={"title":note,"detail":f"Pick pose is simulator-provided (oracle); classifier input is rendered camera RGB crop. Item {seq}/3."}
            self.set_gripper(0.04,0.30,over)
            self.move_and_record(pre,2.00,note,"Approach with simulator-provided position (oracle).")
            self.move_and_record(pick,2.00,note,"Grasp-center IK: align the fingers around the parcel.")
            self.set_gripper(0.018,1.00,over)
            # Contact confirmation before constraint creation; failure is recorded, never fabricated.
            constraint, grasp=self.attach_virtual_grasp(oid)
            lifted=False
            if constraint is not None:
                self.move_and_record(pre,1.65,note,"Virtual grasp constraint enabled after verified finger contact.")
                lift_pos = p.getBasePositionAndOrientation(oid,physicsClientId=self.client)[0]
                lifted = bool(lift_pos[2] > xyz[2] + 0.12)
                grasp['measured_height_after_lift_m'] = float(lift_pos[2])
                self.move_and_record(tray_pre,2.30,note,"Joint-space IK transfer with physics active; AI label determines tray coordinates.")
                self.move_and_record(tray_place,1.25,note,"Lowering parcel into collidable tray; release follows.")
                p.removeConstraint(constraint,physicsClientId=self.client)
                constraint=None
                self.set_gripper(0.04,0.80,{"title":f"RELEASE {predicted} parcel","detail":"Constraint released; parcel must settle inside its matching physical tray."})
                self.move_and_record(tray_pre,1.10,note,"Retreat after release.")
                # Let physical settling complete and verify center is inside tray bounds and z is near tabletop.
                for _ in range(round(1.4/DT)): self._advance({"title":f"SETTLING: {predicted} parcel","detail":"Success is checked from final simulated body position, not animation state."},record=True)
            else:
                # No lift/transport is attempted without real simulated contact; log honest failure.
                self.set_gripper(0.04,0.5,{"title":f"GRASP FAILED: {actual}","detail":"No verified finger contact within the allowed range; no constraint or successful delivery claimed."})
            final=p.getBasePositionAndOrientation(oid,physicsClientId=self.client)[0]
            tx,ty=self.tray_centers[predicted]
            inside=abs(final[0]-tx)<0.066 and abs(final[1]-ty)<0.066 and 0.64 <= final[2] <= 0.69
            success=bool(lifted and inside and predicted==actual)
            row={**base,**grasp,"lifted_after_verified_contact":bool(lifted),"final_xyz_m":[float(v) for v in final],"inside_predicted_tray_bounds":bool(inside),"success":success}
            rows.append(row)
        # Short settling outro; duration is determined by actual simulation steps.
        outro={"title":"RUN COMPLETE | per-parcel result in sorter_log.csv","detail":"Playback/simulated-time ratio: 1.00x (20 video fps; 12 physics steps per frame)."}
        outro['detail']=f'Playback/simulated-time ratio: 1.00x ({FPS} video fps; {RENDER_EVERY} physics steps per frame).'
        for _ in range(round(4.0/DT)): self._advance(outro,record=True)
        return rows

    def close(self):
        if self.writer:
            self.writer.close(); self.writer=None
        if p.isConnected(self.client): p.disconnect(physicsClientId=self.client)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",default=".")
    ap.add_argument("--no-video",action="store_true",help="Run training/evaluation and physics tests without MP4")
    args=ap.parse_args()
    out=Path(args.output).resolve(); out.mkdir(parents=True,exist_ok=True)
    s=Sorter(out,record=not args.no_video)
    summary={"seed":SEED,"backend":"PyBullet DIRECT CPU","video":None,"sort_results":[],"tests":[]}
    try:
        t0=time.time()
        model,metrics=s.fit_and_evaluate(out/"color_mlp.joblib")
        with open(out/"evaluation.json","w",encoding="utf-8") as f: json.dump(metrics,f,indent=2,ensure_ascii=False)
        print(f"TRAIN/EVAL: {metrics['test_correct']}/{metrics['test_total']} = {metrics['test_accuracy']:.3f}; confusion={metrics['confusion_matrix_rows_true_cols_pred']}",flush=True)
        if s.record:
            video=out/"sorter_demo.mp4"
            s.writer=imageio.get_writer(video,fps=FPS,codec="libx264",quality=7,macro_block_size=1,ffmpeg_params=["-pix_fmt","yuv420p","-preset","veryfast"])
        results=s.perform_sort(model)
        summary["sort_results"]=results
        summary["tests"]=[
            {"name":"heldout_render_accuracy_at_least_95_percent","passed":metrics['test_accuracy']>=0.95},
            {"name":"trained_model_predicts_all_three_categories","passed":set(r['predicted_class'] for r in results)==set(LABELS)},
            {"name":"every_success_has_verified_finger_contact","passed":all((not r['success']) or (r['contact_verified'] and r['constraint_created'] and r['lifted_after_verified_contact']) for r in results)},
            {"name":"classifier_prediction_drives_destination","passed":all(r['target_tray']==r['predicted_class'] for r in results)},
            {"name":"per_parcel_delivery","passed":all(r['success'] for r in results)},
        ]
        # Compute true video duration and ratio against simulated time advanced while recording.
        duration=s.frame_count/FPS
        simtime=s.sim_steps*DT
        summary["video"]={"file":"sorter_demo.mp4","width":W,"height":H,"fps":FPS,"frames":s.frame_count,"duration_seconds":duration,"simulated_seconds_advanced":simtime,"playback_to_simulation_time_ratio":duration/simtime if simtime else None,"nominal_ratio":"1.00x (12 physics steps/frame at 240 Hz, 20 fps)","audio":False}
        summary['video'].update({'native_render_width':RENDER_W,'native_render_height':RENDER_H,'nominal_ratio':f'1.00x ({RENDER_EVERY} physics steps/frame at 240 Hz, {FPS} fps)'})
        if not s.record: summary['video'] = None
        if s.record:
            summary['tests'].append({'name':'video_duration_60_to_180_seconds','passed':60<=duration<=180})
            summary['tests'].append({'name':'video_time_matches_simulation_within_one_frame','passed':abs(duration-simtime)<=1/FPS+0.001})
        summary["total_simulation_steps"]=s.sim_steps
        summary["elapsed_wall_seconds_before_summary_write"]=time.time()-t0
        summary["all_sort_success"]=all(r['success'] for r in results)
        summary["all_tests_passed"]=all(t['passed'] for t in summary['tests'])
        with open(out/"run_summary.json","w",encoding="utf-8") as f: json.dump(summary,f,indent=2,ensure_ascii=False)
        with open(out/"sorter_log.csv","w",newline="",encoding="utf-8") as f:
            fields=["seq","actual_class","predicted_class","confidence","target_tray","contact_verified","finger_contact_count","ee_object_distance_m","constraint_created","measured_height_after_lift_m","lifted_after_verified_contact","final_xyz_m","inside_predicted_tray_bounds","success","oracle_pose_used_for_aiming","camera_inferred_pose"]
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
            for r in results: w.writerow({k:(json.dumps(r.get(k),ensure_ascii=False) if isinstance(r.get(k),(dict,list)) else r.get(k)) for k in fields})
        print(f"SORT: {[(r['actual_class'],r['predicted_class'],r['success'],r['contact_verified'],r['final_xyz_m']) for r in results]}",flush=True)
        print(f"TESTS: {summary['tests']}",flush=True)
        print(f"VIDEO: {summary['video']}",flush=True)
        return 0 if summary['all_tests_passed'] else 1
    finally:
        s.close()

if __name__=="__main__": raise SystemExit(main())
