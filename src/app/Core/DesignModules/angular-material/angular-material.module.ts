import {NgModule} from '@angular/core';
import {CommonModule} from '@angular/common';
import {MatInputModule} from "@angular/material/input";
import {MatCardModule} from "@angular/material/card";
import {MatDialogModule} from "@angular/material/dialog";
import {MatButtonModule} from "@angular/material/button";

const AngularMaterial = [
    MatInputModule,
    MatCardModule,
    MatDialogModule,
    MatButtonModule
]

@NgModule({
    exports: [
        AngularMaterial
    ]
})
export class AngularMaterialModule {
}
