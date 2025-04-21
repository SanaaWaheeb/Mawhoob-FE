import {Component, inject} from '@angular/core';
import {MAT_DIALOG_DATA, MatDialogRef} from "@angular/material/dialog";

@Component({
    selector: 'app-statistics',
    standalone: false,
    templateUrl: './statistics.component.html',
    styleUrl: './statistics.component.scss'
})
export class StatisticsComponent {
    readonly dialogRef = inject(MatDialogRef<StatisticsComponent>);
    readonly data = inject<any>(MAT_DIALOG_DATA);
}
