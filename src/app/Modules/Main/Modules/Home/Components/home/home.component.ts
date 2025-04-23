import {Component} from '@angular/core';
import {MatDialog} from "@angular/material/dialog";
import {StatisticsComponent} from "./Components/statistics/statistics.component";
import {ProfileComponent} from "./Components/profile/profile.component";
import {CandidatesComponent} from "./Components/candidates/candidates.component";
import {EntityProfileComponent} from "./Components/entity-profile/entity-profile.component";

@Component({
    selector: 'app-home',
    standalone: false,
    templateUrl: './home.component.html',
    styleUrl: './home.component.scss'
})
export class HomeComponent {
    isPlayer: boolean = false;

    constructor(
        private dialog: MatDialog,
    ) {
    }

    openStatistics(): void {
        this.dialog.open(StatisticsComponent, {
            minWidth: '25vw',
        })
    }

    openProfile(): void {
        this.dialog.open(ProfileComponent, {
            minWidth: '25vw',
        })
    }

    openCandidates(): void {
        this.dialog.open(CandidatesComponent, {
            minWidth: '25vw',
        })
    }
    openEntityProfile(): void {
        this.dialog.open(EntityProfileComponent, {
            minWidth: '25vw',
        })
    }
}
