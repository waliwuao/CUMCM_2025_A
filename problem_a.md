# 2025 Higher Education Press Cup National College Mathematical Modeling Contest

(Please read the "National College Mathematical Modeling Contest Paper Format Specification" first)

# Problem A: Delivery Strategy of Smoke-Screen Decoys

Smoke-screen decoys mainly rely on chemical combustion or explosion to disperse a smoke or aerosol cloud that forms an obscuration in a specific airspace ahead of a target, thereby interfering with incoming enemy missiles. They are low-cost and highly cost-effective. With continuous advances in smoke-screen technology, several delivery modes now exist for the pinpoint scattering of smoke-screen decoys; i.e., the decoy can be controlled precisely to reach a predetermined position before scattering, and its burst time is controlled through a time fuze.

In this problem, long-endurance UAVs (unmanned aerial vehicles) carry a certain type of smoke-screen decoy and loiter in a designated airspace. After receiving a task, a UAV drops the decoy between the incoming weapon and the protected target to create a smoke obscuration. On any single UAV, two decoys must be released at least 1 s apart. After leaving the UAV, the decoy moves under gravity. Upon burst, it instantaneously forms a spherical smoke cloud. Owing to special technology, this cloud descends at a constant speed of 3 m/s. Experimental data show that, within 10 m of the cloud center, the smoke concentration can provide effective obscuration for the target during the first 20 s after burst.

The incoming weapons are air-to-surface missiles flying at 300 m/s. Each missile flies directly toward a decoy that has been specially set up to protect a cylindrical fixed target (radius 7 m, height 10 m). Take the decoy as the origin, the horizontal plane as the x-y plane, and the center of the bottom face of the real target as (0, 200, 0).

When the surveillance radar detects the incoming missiles:
- Missile M1 is at (20 000, 0, 2 000)
- Missile M2 is at (19 000, 600, 2 100)
- Missile M3 is at (18 000, −600, 1 900)

The initial positions of the five UAVs are:
- FY1: (17 800, 0, 1 800)
- FY2: (12 000, 1 400, 1 400)
- FY3: (6 000, −3 000, 700)
- FY4: (11 000, 2 000, 1 800)
- FY5: (13 000, −2 000, 1 300)

During the attack, decoys should be deployed to prevent the missiles from finding the real target. The control center assigns tasks to the UAVs immediately upon radar detection. After accepting its task, a UAV can instantaneously adjust its flight direction, then fly straight and level at a constant speed between 70 m/s and 140 m/s. Each UAV’s heading and speed may differ, but once chosen they remain unchanged.

To achieve better jamming effectiveness, design a delivery strategy for the smoke-screen decoys, covering:
- UAV flight direction
- UAV speed
- Decoy release point
- Decoy burst point

Build a mathematical model and, for each scenario below, design a delivery strategy that maximizes the total effective obscuration time of the real target by multiple decoys. Obscuration periods need not be contiguous.

---

### Problem 1
Using UAV FY1 to drop 1 decoy against M1.  
Given: FY1 flies toward the decoy at 120 m/s, releases 1 decoy 1.5 s after task assignment, and the decoy bursts 3.6 s after release.  
Compute the effective obscuration duration provided by this decoy against M1.

---

### Problem 2
Using UAV FY1 to drop 1 decoy against M1.  
Determine FY1’s flight direction, speed, release point, and burst point so that the obscuration time is maximized.

---

### Problem 3
Using UAV FY1 to drop 3 decoys against M1.  
Give the delivery strategy and save the results to `result1.xlsx` (template attached).

---

### Problem 4
Using UAVs FY1, FY2, FY3, each dropping 1 decoy, against M1.  
Give the delivery strategy and save the results to `result2.xlsx` (template attached).

---

### Problem 5
Using all 5 UAVs, each dropping at most 3 decoys, against missiles M1, M2, M3.  
Give the delivery strategy and save the results to `result3.xlsx` (template attached).
