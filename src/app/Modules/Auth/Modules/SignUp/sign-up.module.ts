import {NgModule} from '@angular/core';
import {CommonModule} from '@angular/common';

import {SignUpRoutingModule} from './sign-up-routing.module';
import {RouterModule} from "@angular/router";
import {SignUpComponent} from './Components/sign-up/sign-up.component';
import {AngularMaterialModule} from "../../../../Core/DesignModules/angular-material/angular-material.module";


@NgModule({
    declarations: [
        SignUpComponent
    ],
    imports: [
        CommonModule,
        SignUpRoutingModule,
        RouterModule,
        AngularMaterialModule
    ]
})
export class SignUpModule {
}
